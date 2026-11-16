"""WebdevReportClient — A2A client for webdev-agent report generation.

Sends a ``ReportSpec`` to the webdev-agent via A2A JSON-RPC ``message/send``,
receives a self-contained HTML report in the response artifacts, and
optionally uploads the HTML to object storage for persistent URL access.

The A2A protocol follows the structure used by ``services/webdev-agent``:
``message/send`` with a text part containing the prompt + ReportSpec JSON.
"""

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from html import escape as _html_escape
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.utils.http_transport import AsyncClient

if TYPE_CHECKING:
    from benchmark.libs.storage_ext.signed_client import ReportStorageClient

logger = get_logger("agent_eval.benchmark.webdev_client")

#: Suspiciously short report-HTML threshold (below this, treat as truncated or an empty shell).
_MIN_REPORT_HTML_LEN = 2000

#: Total generation attempts when LLM output is unusable (empty / not HTML / truncated / JS syntax error),
#: including the first try. A template flaw is a one-off random event; regenerating usually recovers.
_GENERATION_MAX_ATTEMPTS = 2

#: Whole-response rejection phrase for the webdev-agent external-link red line (search_eval-20260914-q3 /
#: ieval_external-20260915 retry incident: a fixed rejection phrase was returned instead of HTML).
_EXTERNAL_LINK_REFUSAL_RE = re.compile(
    r"不允许嵌入用户指定的外部链接|无法完成该请求",
)

#: Default request timeout (seconds).  Report generation involves an LLM call
#: in webdev-agent; one hour covers large specs with many case_results.
_DEFAULT_TIMEOUT_S = 3600.0

#: Presigned report URL TTL (30 days, in seconds) for private-bucket access.
_REPORT_URL_EXPIRES_S = 30 * 24 * 60 * 60

#: Sanitize run_id for the storage object key (prevent path traversal / illegal chars).
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Pattern to locate the <script type="application/json" id="cases-data"> tag
#: in LLM-generated HTML. Used by _fix_case_data to unconditionally replace
#: the cases-data content with original case_results from the spec (PG
#: source of truth), preventing LLM abbreviation of input/output.
_CASES_DATA_PATTERN = re.compile(
    r'<script\s+type="application/json"\s+id="cases-data">\s*(.*?)\s*</script>',
    re.DOTALL,
)

#: Pattern to verify that extracted text is actually HTML (not an LLM error
#: message or truncated response). Accepts optional leading whitespace.
_HTML_PREFIX_RE = re.compile(r"^\s*(?:<!doctype\s+html|<html)", re.IGNORECASE)

#: Pattern to detect an existing <meta charset="..."> declaration.
_CHARSET_RE = re.compile(r"<meta\s+charset\s*=", re.IGNORECASE)

#: Pattern to locate the opening <head> tag for charset injection.
_HEAD_OPEN_RE = re.compile(r"<head[^>]*>", re.IGNORECASE)

#: Fallback: locate the opening <html> tag when no <head> is present.
_HTML_OPEN_RE = re.compile(r"<html[^>]*>", re.IGNORECASE)

#: Query column/detail truncation + exact-search patch (injected before </body>, covering LLM template drift).
#:
#: Key constraint of the exact-search wrap: the LLM template's filter-result variable name is not fixed
#: (filtered / filteredCases / ...), and some templates already include exact-search logic.
#: So wrapped does not clear the search box (it keeps origApply's own search result), and detects the
#: result variable that was reassigned by comparing reference snapshots before and after origApply.
#: It then applies an idempotent exact filter (an already-exact result is unchanged) and writes back every
#: candidate variable. A name mismatch drops the search result and the box stops working entirely
#: (regression in the ieval_external-20260826 report).
#:
#: Other deterministic fallbacks (covering LLM template drift):
#: - The dimension filter (the "all dimensions" dropdown) is removed per spec: hide it and reset to the
#:   default option rather than deleting the node, so template JS can still read the element safely;
#: - fullTextOf extracts the full query text. A template cell often has both a truncated element and a
#:   full-text element, and reading textContent concatenates them into garbage (ieval_external-20260826:
#:   the user copied garbage from the hover layer, searched, got 0 results, and no case was clickable).
#:   Prefer the longest text from the full-text container. That container's class drifts with the LLM template:
#:   besides .query-overlay there have been .query-hover (search_eval-20260909 report: the hover layer missed
#:   the selector, so the textContent fallback concatenated the visible truncated text with the hidden full
#:   text into a doubled query) and .hover-overlay (ieval_external-20260916).
#:   All three names must be covered;
#: - When exact search returns 0 and the input ends with …/..., a page truncation marker, fall back to
#:   a display_input prefix match, so "copy the visible truncated text, then search" can hit.
#: - A template's own .cell-truncate > .hover-overlay is often clipped by overflow:hidden
#:   (ieval_external-20260916: query/output hover has no full text). The delegated listener must
#:   cover .cell-truncate and switch to a position:fixed floating layer.
_QUERY_UX_PATCH = """
<style id="query-ux-patch">
.query-clip{display:inline-block;max-width:280px;vertical-align:bottom;cursor:pointer;color:#4f46e5}
.query-toggle{cursor:pointer;color:#4f46e5}
.trace-detail-row>td{background:#f8f9fb;padding:0;border-bottom:1px solid #e2e4ea}
.trace-dropdown{padding:16px 20px;border-left:3px solid #4f46e5}
.query-short{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.query-tip-float{position:fixed;z-index:9999;max-width:min(480px,80vw);max-height:240px;
overflow:auto;padding:8px 10px;background:#111827;color:#f9fafb;font-size:12px;
line-height:1.5;white-space:pre-wrap;user-select:text;-webkit-user-select:text;
border-radius:6px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
</style>
<script id="query-ux-patch-js">
(function(){
  var TABLE_MAX=40,MODAL_MAX=80,tipEl=null,hideTimer=null;
  // LLM 常把 Case 表明成 class=case-tbl / tbody#case-tbody，漏 id=data-table
  // （ieval_external-20260916：补丁按 #data-table 早退 → 悬浮/点击/Trace 全失效）。
  function findCaseTable(){
    var table=document.getElementById('data-table');
    if(table) return table;
    var tbody=document.getElementById('case-tbody');
    if(tbody){
      table=tbody.parentElement;
      while(table&&table.tagName!=='TABLE') table=table.parentElement;
      if(table){ table.id='data-table'; return table; }
    }
    var cands=document.querySelectorAll(
      'table.case-tbl,#table-wrapper table,.table-wrapper table');
    var i,j,ths,t;
    for(i=0;i<cands.length;i++){
      ths=cands[i].querySelectorAll('thead th');
      for(j=0;j<ths.length;j++){
        t=String(ths[j].textContent||'').replace(/\\s+/g,'');
        if(t==='query'||t==='用例ID'){ cands[i].id='data-table'; return cands[i]; }
      }
    }
    ths=document.querySelectorAll('th');
    for(i=0;i<ths.length;i++){
      t=String(ths[i].textContent||'').replace(/\\s+/g,'');
      if(t!=='query'&&t!=='用例ID') continue;
      table=ths[i];
      while(table&&table.tagName!=='TABLE') table=table.parentElement;
      if(table){ table.id='data-table'; return table; }
    }
    return null;
  }
  function clip(t,n){t=String(t||'');return t.length>n?t.slice(0,n)+'\\u2026':t;}
  function htmlFor(text,max){
    text=String(text||'').trim();
    if(!text) return '<span class="muted">-</span>';
    return '<span class="query-clip query-toggle" tabindex="0" data-full="'+
      text.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;')+
      '"><span class="query-short"></span></span>';
  }
  function fillShort(root,max){
    var nodes=root.querySelectorAll('.query-clip[data-full]');
    for(var i=0;i<nodes.length;i++){
      var el=nodes[i],full=el.getAttribute('data-full')||'';
      var s=el.querySelector('.query-short');
      if(s) s.textContent=clip(full,max);
    }
  }
  function fullTextOf(el){
    var own=el.getAttribute('data-full');
    if(own!=null&&String(own).trim()) return String(own).trim();
    var holders=el.querySelectorAll('.query-overlay,.query-full,.detail-query-overlay,'+
      '.query-hover,.detail-query-hover,.hover-overlay,[data-full]');
    var best='',i,t;
    for(i=0;i<holders.length;i++){
      var df=holders[i].getAttribute('data-full');
      t=String(df!=null?df:(holders[i].textContent||'')).trim();
      if(t.length>best.length) best=t;
    }
    if(best) return best;
    var kids=el.children;
    for(i=0;i<kids.length;i++){
      t=String(kids[i].textContent||'').trim();
      if(t.length>best.length) best=t;
    }
    if(best) return best;
    return String(el.textContent||'').trim();
  }
  function neutralizeDimFilter(){
    var dimSel=document.querySelector('select#dim-filter');
    if(!dimSel){
      var sels=document.querySelectorAll('select');
      for(var si=0;si<sels.length;si++){
        var opts=sels[si].options;
        for(var oi=0;oi<opts.length;oi++){
          if((opts[oi].textContent||'').trim()==='全部维度'){dimSel=sels[si];break;}
        }
        if(dimSel)break;
      }
    }
    if(dimSel&&dimSel.style.display!=='none'){
      dimSel.style.display='none';
      for(var di=0;di<dimSel.options.length;di++){
        if((dimSel.options[di].textContent||'').trim()==='全部维度'){dimSel.selectedIndex=di;break;}
      }
    }
  }
  function hideTip(){if(tipEl&&tipEl.parentNode)tipEl.parentNode.removeChild(tipEl);tipEl=null;}
  function scheduleHide(){clearTimeout(hideTimer);hideTimer=setTimeout(hideTip,280);}
  function cancelHide(){clearTimeout(hideTimer);}
  function showTip(text,anchor){
    hideTip();
    tipEl=document.createElement('div');
    tipEl.className='query-tip-float';
    tipEl.textContent=text;
    document.body.appendChild(tipEl);
    var r=anchor.getBoundingClientRect();
    var left=Math.max(8,Math.min(r.left,window.innerWidth-Math.min(480,window.innerWidth*0.8)-16));
    tipEl.style.left=left+'px';
    tipEl.style.top=(r.bottom+6)+'px';
    tipEl.addEventListener('mouseenter',cancelHide);
    tipEl.addEventListener('mouseleave',scheduleHide);
  }
  document.addEventListener('mouseover',function(e){
    var clipEl=e.target&&e.target.closest&&
      e.target.closest('.query-clip[data-full],.output-clip[data-full],.case-id-link[data-full],.cell-truncate');
    if(!clipEl) return;
    cancelHide();
    var full=clipEl.getAttribute('data-full')||'';
    if(!full) full=fullTextOf(clipEl);
    if(full) showTip(full,clipEl);
  });
  document.addEventListener('mouseout',function(e){
    var clipEl=e.target&&e.target.closest&&
      e.target.closest('.query-clip,.output-clip,.case-id-link,.cell-truncate');
    if(!clipEl) return;
    var rel=e.relatedTarget;
    if(rel&&(clipEl.contains(rel)||(tipEl&&tipEl.contains(rel)))) return;
    scheduleHide();
  });
  function queryColIndex(table){
    var ths=table.querySelectorAll('thead th');
    for(var i=0;i<ths.length;i++){
      if(/^\\s*query\\s*$/i.test(ths[i].textContent||'')) return i;
    }
    return 1;
  }
  function enhanceTable(){
    neutralizeDimFilter();
    var table=findCaseTable();
    if(!table) return;
    var qi=queryColIndex(table);
    var rows=table.querySelectorAll('tbody tr');
    for(var r=0;r<rows.length;r++){
      var td=rows[r].children[qi];
      if(!td||td.querySelector('.query-clip')) continue;
      var full=fullTextOf(td);
      if(!full||full==='-') continue;
      td.innerHTML=htmlFor(full,TABLE_MAX);
      // 模板可能给 td 自带 data-full（其 setupCellHovers 会再叠一层
      // hover-tip）——移除，悬浮全文统一走本补丁的 query-tip-float。
      td.removeAttribute('data-full');
    }
    fillShort(table,TABLE_MAX);
  }
  function enhanceModal(){
    var modal=document.querySelector('.detail-modal');
    if(!modal) return;
    var labeled=modal.querySelectorAll('[data-query],.detail-query,.query-full');
    var i,full,el;
    for(i=0;i<labeled.length;i++){
      el=labeled[i];
      if(el.querySelector('.query-clip')) continue;
      full=fullTextOf(el);
      if(full&&full!=='-') el.innerHTML=htmlFor(full,MODAL_MAX);
    }
    var labels=modal.querySelectorAll('dt,.label,.detail-label,strong,b');
    for(i=0;i<labels.length;i++){
      var t=(labels[i].textContent||'').replace(/[:：]\\s*$/,'').trim().toLowerCase();
      if(t!=='query'&&t!=='用户query') continue;
      var val=labels[i].nextElementSibling;
      if(!val||val.querySelector('.query-clip')) continue;
      full=fullTextOf(val);
      if(full&&full!=='-') val.innerHTML=htmlFor(full,MODAL_MAX);
    }
    fillShort(modal,MODAL_MAX);
  }
  function ensureTraceToggle(){
    // 统一静态模板已带 buildBasicInfoHtml + 完整 traceDetailHtml，勿覆盖。
    if(typeof window.buildBasicInfoHtml==='function') return;
    if(typeof window.toggleTraceDetail==='function'&&window.toggleTraceDetail.__richDetail) return;
    window.toggleTraceDetail=function(el,key){
      var td=el&&el.closest&&el.closest('td');
      var row=td&&td.parentElement;
      if(!row) return;
      hideTip();
      var next=row.nextElementSibling;
      if(next&&next.classList&&next.classList.contains('trace-detail-row')){
        next.parentNode.removeChild(next);
        if(el) el.classList.remove('open');
        return;
      }
      var openRows=document.querySelectorAll('.trace-detail-row');
      var oi;
      for(oi=0;oi<openRows.length;oi++){
        if(openRows[oi].parentNode) openRows[oi].parentNode.removeChild(openRows[oi]);
      }
      var opens=document.querySelectorAll('.query-toggle.open');
      for(oi=0;oi<opens.length;oi++) opens[oi].classList.remove('open');
      var cases=window.CASES||[];
      var c=null,k=String(key||''),ci;
      for(ci=0;ci<cases.length;ci++){
        if(String(cases[ci].case_id||'')===k||String(cases[ci].trace_id||'')===k||
           String(cases[ci].display_case_id||'')===k){ c=cases[ci]; break; }
      }
      if(!c){
        var qn=String((el&&el.getAttribute('data-full'))||'').trim();
        if(qn){
          for(ci=0;ci<cases.length;ci++){
            if(String(cases[ci].display_input||'').trim()===qn){ c=cases[ci]; break; }
          }
        }
      }
      if(!c){
        var table=findCaseTable();
        var bodyRows=table?table.querySelectorAll('tbody tr:not(.trace-detail-row)'):[];
        var idx=-1,ri;
        for(ri=0;ri<bodyRows.length;ri++){ if(bodyRows[ri]===row){ idx=ri; break; } }
        var page=Math.max(1,Number(window.currentPage)||1);
        var filtered=Array.isArray(window.filteredCases)?window.filteredCases:cases;
        c=filtered[(page-1)*10+idx]||null;
      }
      if(!c) return;
      // openDetail 包装器（injectTraceLink）以 window.__ruxActiveTraceUrl 为唯
      // 一数据源；query 下拉点击路径同步写入，保证模板其它入口触发 openDetail
      // 时模态框头部仍可注入 Langfuse 外链（否则 !url 早退、永不注入）。
      window.__ruxActiveTraceUrl=String(c.trace_url||'').trim();
      function esc(s){
        return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
          .replace(/"/g,'&quot;');
      }
      function gradeColor(g){
        return g==='EXCELLENT'||g==='GOOD'?'#16a34a'
          :g==='ACCEPTABLE'?'#2563eb':'#dc2626';
      }
      var did=String(c.trace_id||c.case_id||c.display_case_id||'-');
      var grade=String(c.grade||'');
      var fused=(c.scores&&c.scores.fused&&c.scores.fused.score!=null)
        ?Number(c.scores.fused.score).toFixed(3):'-';
      var turl=String(c.trace_url||'').trim();
      var gc=grade?gradeColor(grade):'';
      var gradeDd=grade
        ?('<dt>case状态</dt><dd><span class="grade-badge" style="background:'
          +gc+'20;color:'+gc+'">'+esc(grade)+'</span></dd>')
        :'<dt>case状态</dt><dd class="muted">-</dd>';
      // 优先复用统一模板 pickJudgeReason；webdev LLM HTML 无该函数时按
      // REPORT.judge_reason_preferred_keys（插件透出）本地兜底，避免静默空白。
      var report=window.REPORT||{};
      var judgeReason='';
      if(typeof window.pickJudgeReason==='function'){
        judgeReason=String(window.pickJudgeReason(c)||'');
      }else if(c.scores&&typeof c.scores==='object'){
        var pref=Array.isArray(report.judge_reason_preferred_keys)
          &&report.judge_reason_preferred_keys.length
          ? report.judge_reason_preferred_keys : ['fused','综合质量'];
        var pi, psc, pr;
        for(pi=0;pi<pref.length;pi++){
          psc=c.scores[pref[pi]];
          if(psc&&typeof psc==='object'){
            pr=String(psc.reason||'').trim();
            if(pr){ judgeReason=pr; break; }
          }
        }
        if(!judgeReason){
          var sk=Object.keys(c.scores), si, ssc, sr;
          for(si=0;si<sk.length;si++){
            if(sk[si]==='fused') continue;
            ssc=c.scores[sk[si]];
            if(!ssc||typeof ssc!=='object') continue;
            sr=String(ssc.reason||'').trim();
            if(sr.length>judgeReason.length) judgeReason=sr;
          }
        }
      }
      var reasonDd=judgeReason
        ?('<dt>评分理由</dt><dd style="white-space:pre-wrap;line-height:1.55">'
          +esc(judgeReason)+'</dd>')
        :'';
      var traceDd=turl
        ?('<dt>Trace链接</dt><dd><a class="trace-link" href="'+esc(turl)
          +'" target="_blank" rel="noopener">Trace ↗</a></dd>')
        :'<dt>Trace链接</dt><dd class="muted">-</dd>';
      var basic='<div class="detail-section">'
        +'<div class="detail-section-title">基本信息</div><dl class="detail-kv">'
        +'<dt>用例ID</dt><dd>'+esc(did)+'</dd>'
        +gradeDd
        +'<dt>综合得分</dt><dd style="font-weight:700;font-size:16px">'
        +fused+'</dd>'
        +reasonDd
        +traceDd
        +'</dl></div>';
      var q=String(c.display_input||'').trim();
      var result=String(c.display_output||'').trim();
      var outHtml=(result&&typeof window.renderMarkdown==='function')
        ? window.renderMarkdown(result)
        : '<div class="detail-text">'+(result?esc(result):'-')+'</div>';
      var criteria=Array.isArray(report.criteria)?report.criteria:[];
      var dimHtml='';
      if(criteria.length&&c.scores){
        var rows='';
        for(var di=0;di<criteria.length;di++){
          var cr=criteria[di];
          var sc=c.scores[cr.id];
          if(!sc) continue;
          var sv=sc.score!=null?Number(sc.score).toFixed(3):'—';
          var reason=esc(sc.reason||'');
          rows+='<div class="detail-score-row">'
            +'<span class="detail-score-name">'+esc(cr.name)+'</span>'
            +'<span class="detail-score-val">'+sv+'</span>'
            +'<span class="detail-score-reason">'+reason+'</span></div>';
          if(sc.judge){
            rows+='<div style="padding:2px 0 4px 130px;font-size:11px;color:#6b7194">'
              +esc(sc.judge.triage_level||'')+' '
              +esc(sc.judge.triage_raw||'')+'</div>';
          }
        }
        if(rows){
          dimHtml='<div class="detail-section">'
            +'<div class="detail-section-title">各维度评分</div>'
            +rows+'</div>';
        }
      }
      var attrHtml='';
      var a=c.attribution;
      if(a&&typeof a==='object'){
        attrHtml='<div class="detail-section">'
          +'<div class="detail-section-title">用例归因</div>'
          +'<dl class="detail-kv">';
        if(a.primary_issue_layer){
          attrHtml+='<dt>问题层</dt><dd>'+esc(a.primary_issue_layer)+'</dd>';
        }
        if(a.primary_issue_detail){
          attrHtml+='<dt>归因推理</dt><dd style="white-space:pre-wrap">'
            +esc(a.primary_issue_detail)+'</dd>';
        }
        if(a.severity){
          attrHtml+='<dt>严重程度</dt><dd>'+esc(a.severity)+'</dd>';
        }
        if(a.root_cause_chain){
          var chain=typeof a.root_cause_chain==='string'
            ?a.root_cause_chain:JSON.stringify(a.root_cause_chain);
          attrHtml+='<dt>根因链</dt><dd style="white-space:pre-wrap">'
            +esc(chain)+'</dd>';
        }
        if(a.improvement_suggestion){
          attrHtml+='<dt>改进建议</dt><dd style="white-space:pre-wrap">'
            +esc(a.improvement_suggestion)+'</dd>';
        }
        if(a.confidence!=null){
          attrHtml+='<dt>置信度</dt><dd>'
            +(Number(a.confidence)*100).toFixed(0)+'%</dd>';
        }
        if(a.triage_level){
          attrHtml+='<dt>分级</dt><dd>'+esc(a.triage_level)+'</dd>';
        }
        if(a.root_cause){
          attrHtml+='<dt>根因</dt><dd>'+esc(a.root_cause)+'</dd>';
        }
        if(a.summary){
          attrHtml+='<dt>摘要</dt><dd style="white-space:pre-wrap">'
            +esc(a.summary)+'</dd>';
        }
        attrHtml+='</dl></div>';
      }
      var detail=document.createElement('tr');
      detail.className='trace-detail-row';
      var cell=document.createElement('td');
      cell.colSpan=row.children.length||8;
      cell.innerHTML='<div class="trace-dropdown">'
        +basic
        +'<div class="detail-section"><div class="detail-section-title">输入</div>'
        +'<div class="detail-text">'+(q?esc(q):'-')+'</div></div>'
        +'<div class="detail-section"><div class="detail-section-title">输出</div>'
        +outHtml+'</div>'
        +dimHtml
        +attrHtml
        +'</div>';
      detail.appendChild(cell);
      if(row.parentNode) row.parentNode.insertBefore(detail,row.nextSibling);
      if(el) el.classList.add('open');
    };
    window.toggleTraceDetail.__richDetail=true;
  }
  function wrap(){
    var boxes=document.querySelectorAll('.search-box');
    var b,i;
    for(i=0;i<boxes.length;i++) boxes[i].placeholder='精确搜索用例ID / query';
    var origApply=window.applyFilters;
    if(typeof origApply==='function'&&!origApply.__queryExact){
      var wrapped=function(){
        var box=document.querySelector('.search-box')||
          document.querySelector('input[type="search"]');
        var q=box?String(box.value||'').trim():'';
        var NAMES=['filtered','filteredCases','filteredData','filteredList','filteredRows'];
        var before={},i,nm,v;
        for(i=0;i<NAMES.length;i++){
          try{before[NAMES[i]]=window[NAMES[i]];}catch(e){}
        }
        origApply();
        if(q){
          var src=null,found=[];
          for(i=0;i<NAMES.length;i++){
            nm=NAMES[i];
            v=window[nm];
            if(Object.prototype.toString.call(v)==='[object Array]'){
              found.push(nm);
              if(src===null&&before[nm]!==v) src=v;
            }
          }
          if(src===null&&found.length) src=window[found[0]];
          if(src===null) src=window.CASES||[];
          var next=src.filter(function(c){
            var ids=[c.trace_id,c.case_id,c.display_case_id];
            for(var k=0;k<ids.length;k++){ if(String(ids[k]||'').trim()===q) return true; }
            return String(c.display_input||'').trim()===q;
          });
          if(!next.length&&/(?:…|\\.\\.\\.)$/.test(q)){
            var qp=q.replace(/(?:…|\\.\\.\\.)+$/,'');
            var all=window.CASES||[];
            if(qp){
              next=all.filter(function(c){
                return String(c.display_input||'').trim().lastIndexOf(qp,0)===0;
              });
            }
          }
          for(i=0;i<found.length;i++){ try{window[found[i]]=next;}catch(e){} }
          window.filtered=next;
          if(typeof renderTable==='function') renderTable();
          if(typeof renderPagination==='function') renderPagination();
        }
        enhanceTable();
      };
      wrapped.__queryExact=true;
      window.applyFilters=wrapped;
    }
    var origOpen=window.openDetail;
    if(typeof origOpen==='function'&&!origOpen.__queryClip){
      var wrappedOpen=function(){
        origOpen.apply(this,arguments);
        setTimeout(function(){
          enhanceModal();
          var modal=document.querySelector('#detail-modal,.detail-modal');
          if(!modal) return;
          var labels=modal.querySelectorAll('dt,.label,.detail-label,th,strong,b');
          var li,t,sib;
          for(li=labels.length-1;li>=0;li--){
            t=(labels[li].textContent||'').replace(/[:：]\\s*$/,'').trim().toLowerCase();
            if(t!=='用户id'&&t!=='userid'&&t!=='user_id'&&t!=='uid'&&t!=='adiu') continue;
            sib=labels[li].nextElementSibling;
            if(sib&&sib.parentNode) sib.parentNode.removeChild(sib);
            if(labels[li].parentNode) labels[li].parentNode.removeChild(labels[li]);
          }
          var titles=modal.querySelectorAll('.detail-section-title');
          var ti,lab,sec;
          for(ti=0;ti<titles.length;ti++){
            lab=(titles[ti].textContent||'').trim();
            if(lab!=='输入'&&lab!=='输出') continue;
            sec=titles[ti].parentNode;
            if(sec) sec.style.display='none';
          }
        },0);
      };
      wrappedOpen.__queryClip=true;
      window.openDetail=wrappedOpen;
    }
    ensureTraceToggle();
    if(!document.__queryTraceBind){
      document.__queryTraceBind=true;
      document.addEventListener('click',function(e){
        var el=e.target&&e.target.closest&&
          e.target.closest('.query-clip,.query-toggle');
        if(!el) return;
        if(el.closest('.detail-modal,#detail-modal,.trace-dropdown')) return;
        e.preventDefault();
        e.stopPropagation();
        hideTip();
        var key=el.getAttribute('data-open-key')||el.getAttribute('data-full')||'';
        if(typeof toggleTraceDetail==='function') toggleTraceDetail(el,key);
      });
    }
    var origRender=window.renderTable;
    if(typeof origRender==='function'&&!origRender.__queryClip){
      var wrappedRender=function(){
        // 模板原版抛错（如行绑定闭包 bug）不得阻断增强链——try 容错后
        // enhanceTable 必执行（ieval_external-20260916：TypeError 穿透
        // 全部包装器 → 悬浮/点击补丁被整体跳过）。
        try{ origRender.apply(this,arguments); }catch(e){
          console.error('query-ux renderTable orig',e); }
        enhanceTable();
      };
      wrappedRender.__queryClip=true;
      window.renderTable=wrappedRender;
    }
    enhanceTable();
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',wrap);
  else wrap();
})();
</script>
"""

#: Case-detail column reshape (case-id truncation jumps to the trace / output column after query / drop the action column)
#: + overall-attribution table (Chinese labels) + default FAIL/POOR filter (injected before </body>, covering
#: LLM template drift).
#:
#: Background (ieval_external-20260915 user feedback):
#: - The first Case-detail column becomes "case id" (value=trace_id, click opens the detail modal), and the
#:   output column is dropped. The LLM may still emit the old whitelist table; this patch reshapes it.
#: - The overall-attribution primary_issue_category distribution was a horizontal dist-row bar chart with
#:   English labels (planning/strategy_error); it becomes a Chinese-label table (category|bar|count|percent).
#: - Case details show only FAIL+POOR by default; other tiers appear only when chosen in the status dropdown.
#: - Clicking overall attribution filtered to an empty table: stats map an empty category to "unknown", while
#:   the Case side stays "". The default FAIL+POOR scale also disagrees with the full attribution scale.
#:   Normalize unknown while filtering, and switch to "all statuses" when a category is clicked.
#: - search_eval (20260916): the fused field is the deduction dimension (key plus display name), and the case
#:   has no primary_issue_category. After a click, match dimensions that are not full score; do not force ieval.
#: - search_eval-20260916 (phase 2): the template named the status dropdown id grade-filter
#:   (not case-grade-filter), so setupDefaultFilter returned early and broke the chain (the applyFilters /
#:   filterByRootCause / renderTable wraps all failed: overall-attribution clicks showed an empty table,
#:   case-id column reshape was lost, and output-clip did not apply). Look up the id defensively, and a
#:   missing sel no longer blocks the wrap. Also drop the template's row-level onclick (only "case id" is
#:   clickable), make openDetail accept either a numeric index or a caseId, and remove td data-full
#:   (avoids stacking the template hover-tip and query-tip-float).
#: - ieval_external-20260916-a1: the template wrote the "all statuses" option as value="ALL"
#:   (truthy), so the grade filter `gv && c.grade!==gv` was always true and Case details stayed at 0 rows.
#:   The filter must exclude 'ALL'. When attribution switches to all statuses, an unmatched empty value
#:   falls back to selectedIndex=0 so the dropdown is not blank.
#: - ieval_external-20260916: the case id is truncated to "first 3...last 3", data-full hover shows the
#:   full trace id, and a click opens the detail modal on this page (no new tab).
#:   The output column returns after the query column (display_output, truncated to 40 chars, may wrap,
#:   hover shows the full text). That day's production table had only class=case-tbl / tbody#case-tbody
#:   and no id=data-table, so the patch returned early on the id: case id had no hover/click, and the Trace
#:   link column stayed at the '-' placeholder (trace_url in cases-data actually had a value). "Eval
#:   dimensions" are merged into one compact card (narrower table on the left + centered radar + forced
#:   legend), and renderRadar no longer stacks canvases on redraw or drops radar-legend when cards merge.
_REPORT_UX_PATCH = """
<style id="report-ux-patch">
.case-id-link{color:#4f46e5;text-decoration:underline;cursor:pointer;font-weight:600;
white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.case-id-link:hover{color:#3730a3}
.rux-trace-link{color:#4f46e5;text-decoration:underline;font-size:12px;margin-left:8px;
white-space:nowrap}
.output-clip{display:inline-block;width:10em;max-width:300px;vertical-align:top;cursor:text}
.output-short{display:block;white-space:normal;word-break:break-word;line-height:1.5}
.attr-tbl{width:100%;border-collapse:collapse;font-size:0.88em;margin:8px 0 4px}
.attr-tbl tr{display:table-row}
.attr-tbl td{display:table-cell}
.attr-tbl th{padding:7px 10px;color:#6b7194;font-weight:600;font-size:0.92em;
text-align:left;background:#f8f9fb;border-bottom:1px solid #e2e4ea}
.attr-tbl td{padding:7px 10px;border-bottom:1px solid #eef0f4;vertical-align:middle}
.attr-tbl .ac-cat{color:#374151;font-weight:600;white-space:nowrap}
.attr-tbl .ac-bar{width:42%;}
.attr-tbl .ac-track{display:block;height:10px;background:#eef0f4;border-radius:5px;
overflow:hidden;width:100%}
.attr-tbl .ac-fill{height:100%;background:#4f46e5;border-radius:5px}
.attr-tbl .ac-cnt{width:12%;text-align:right;color:#374151;font-weight:600;white-space:nowrap}
.attr-tbl .ac-pct{width:12%;text-align:right;color:#6b7194;white-space:nowrap}
/* search_eval 模板 .dist-bar-fill 常缺 background（宽有色无）→ 进度条不可见 */
.dist-bar-fill,.attr-bar-fill{background:#4f46e5}
.grade-fp{color:#dc2626}
/* 维度得分明细：维度列按内容收窄，与权重列间距 ≤ 3ch */
#dim-table-card table{width:auto;max-width:100%;border-collapse:collapse}
#dim-table-card th,#dim-table-card td{padding-top:5px;padding-bottom:5px;font-size:12px}
.rux-dim-name{white-space:nowrap;width:1%;padding-right:3ch!important}
.rux-dim-weight{white-space:nowrap;width:1%;padding-left:0!important}
.radar-legend{display:flex;justify-content:center;align-items:center;gap:16px;
margin-top:8px;font-size:12px;color:#6b7194}
.radar-legend .rux-leg{display:flex;align-items:center;gap:6px;cursor:pointer;
user-select:none}
.radar-legend .rux-leg.off{opacity:.35}
.radar-legend .rux-leg.off .rux-leg-txt{text-decoration:line-through}
.radar-legend .rux-sw{display:inline-block;width:28px;height:3px;border-radius:1px}
.radar-legend .rux-sw.today{background:#4f46e5}
.radar-legend .rux-sw.yesterday{
background:repeating-linear-gradient(90deg,#d97706 0 6px,transparent 6px 10px)}
</style>
<script id="report-ux-patch-js">
(function(){
  function findCaseTable(){
    var table=document.getElementById('data-table');
    if(table) return table;
    var tbody=document.getElementById('case-tbody');
    if(tbody){
      table=tbody.parentElement;
      while(table&&table.tagName!=='TABLE') table=table.parentElement;
      if(table){ table.id='data-table'; return table; }
    }
    var cands=document.querySelectorAll(
      'table.case-tbl,#table-wrapper table,.table-wrapper table');
    var i,j,ths,t;
    for(i=0;i<cands.length;i++){
      ths=cands[i].querySelectorAll('thead th');
      for(j=0;j<ths.length;j++){
        t=String(ths[j].textContent||'').replace(/\\s+/g,'');
        if(t==='query'||t==='用例ID'){ cands[i].id='data-table'; return cands[i]; }
      }
    }
    ths=document.querySelectorAll('th');
    for(i=0;i<ths.length;i++){
      t=String(ths[i].textContent||'').replace(/\\s+/g,'');
      if(t!=='query'&&t!=='用例ID') continue;
      table=ths[i];
      while(table&&table.tagName!=='TABLE') table=table.parentElement;
      if(table){ table.id='data-table'; return table; }
    }
    return null;
  }
  // primary_issue_category 英文标签 → 中文（ieval 等业务已知分类）
  var CN_MAP={
    'planning/strategy_error':'规划/策略错误',
    'planning/intent_error':'规划/意图错误',
    'planning/context_lost':'规划/上下文丢失',
    'tool/tool_exec_error':'工具/执行错误',
    'tool/tool_unavailable':'工具/不可用',
    'tool/data_insufficient':'工具/数据不足',
    'generation/factuality_error':'生成/事实性错误',
    'generation/completeness_error':'生成/完整性错误'
  };
  function cnLabel(s){ s=String(s||''); return CN_MAP[s]||s; }
  function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;')
    .replace(/</g,'&lt;').replace(/>/g,'&gt;');}
  function colIndex(table,name){
    var ths=table.querySelectorAll('thead th');
    for(var i=0;i<ths.length;i++){
      if(new RegExp('^\\\\s*'+name+'\\\\s*$','i').test(ths[i].textContent||'')) return i;
    }
    return -1;
  }
  function caseIdFromRow(row){
    var btn=row.querySelector('[onclick*="openDetail"]');
    if(!btn) return '';
    var m=String(btn.getAttribute('onclick')||'').match(/openDetail\\(\\s*['"]([^'"]+)['"]/);
    return m?m[1]:'';
  }
  function findCaseById(cases,id){
    if(!id) return null;
    for(var i=0;i<cases.length;i++){
      var c=cases[i];
      var ids=[c.case_id,c.display_case_id,c.trace_id];
      for(var j=0;j<ids.length;j++){
        if(ids[j]!=null&&String(ids[j])===String(id)) return c;
      }
    }
    var m=/^(.{3})\\.\\.\\.(.{3})$/.exec(String(id));
    if(m){
      for(var k=0;k<cases.length;k++){
        var c2=cases[k];
        var ids2=[c2.case_id,c2.display_case_id,c2.trace_id];
        for(var n=0;n<ids2.length;n++){
          var s=ids2[n]==null?'':String(ids2[n]);
          if(s.length>6&&s.slice(0,3)===m[1]&&s.slice(-3)===m[2]) return c2;
        }
      }
    }
    return null;
  }
  function removeColByName(table,name){
    var idx=colIndex(table,name);
    if(idx<0) return;
    var ths=table.querySelectorAll('thead th');
    if(ths[idx]&&ths[idx].parentNode) ths[idx].parentNode.removeChild(ths[idx]);
    var rows=table.querySelectorAll('tbody tr');
    for(var r=0;r<rows.length;r++){
      var td=rows[r].children[idx];
      if(td) rows[r].removeChild(td);
    }
  }
  // fused 归因字段：ieval=primary_issue_category；search_eval=扣分维度
  function fusedAttrField(){
    var am=window.REPORT&&REPORT.attribution_metrics;
    var entry=am&&am.fused;
    var fds=entry&&entry.field_distributions;
    if(!fds||typeof fds!=='object') return 'primary_issue_category';
    if(fds['扣分维度']) return '扣分维度';
    if(fds['primary_issue_category']) return 'primary_issue_category';
    var keys=Object.keys(fds);
    return keys.length?keys[0]:'primary_issue_category';
  }
  // ── #1 Case 明细：用例ID 截断跳 trace + query 后 output 列 + 去操作列 ──
  function clipText(t,n){ t=String(t||''); return t.length>n?t.slice(0,n)+'\u2026':t; }
  // output 列文本：优先 display_output（调用方预提取的 result 文本）；
  // 缺失时回退解析 output JSON 的 result/response_text/text 字段。
  function outputTextOf(c){
    if(!c) return '';
    var d=String(c.display_output||'').trim();
    if(d) return d;
    var raw=c.output;
    if(raw==null) return '';
    var s=String(raw).trim();
    if(!s) return '';
    if(s.charAt(0)==='{'){
      try{
        var o=JSON.parse(s);
        var r=o&&(o.result||o.response_text||o.text);
        if(r) return String(r);
      }catch(e){}
    }
    return s;
  }
  function reshapeCaseColumns(){
    var table=findCaseTable();
    if(!table) return;
    var theadRow=table.querySelector('thead tr');
    if(!theadRow) return;
    // Case 明细不再展示「用例ID」列：详情入口改由 query 下拉承载。
    removeColByName(table,'用例ID');
    // output 列跟在 query 列后：模板缺列时插入（表头 + 现有每行单元格）。
    // 模板每次 renderTable 重建 tbody——新行没有我们插入的单元格，行循环里
    // 按列数差补齐（insertedOutput 标记仅在我们插入列时才需要补）。
    var qi=colIndex(table,'query');
    var oi=qi>=0?colIndex(table,'output'):-1;
    var insertedOutput=false;
    if(qi>=0&&oi<0){
      var oth=document.createElement('th');
      oth.textContent='output';
      theadRow.insertBefore(oth, theadRow.children[qi+1]||null);
      var orows=table.querySelectorAll('tbody tr');
      for(var ri=0;ri<orows.length;ri++){
        var otd=document.createElement('td');
        orows[ri].insertBefore(otd, orows[ri].children[qi+1]||null);
      }
      oi=qi+1;
      insertedOutput=true;
    }
    var cases=window.CASES||[];
    var bodyRows=table.querySelectorAll('tbody tr');
    var filtered=Array.isArray(window.filteredCases)?window.filteredCases:null;
    var page=Math.max(1,Number(window.currentPage)||1);
    var pageOffset=(page-1)*10;
    var thCount=theadRow.children.length;
    // 行级点击守卫（捕获阶段）：模板若用 addEventListener 绑整行 click
    // （removeAttribute 对其无效），在 tbody 捕获阶段拦截非入口点击，
    // 阻断「整行任意列弹详情」。放行：query 下拉、Trace 外链。
    var tbodyEl=table.querySelector('tbody');
    if(tbodyEl&&!tbodyEl.__ruxRowGuard){
      tbodyEl.__ruxRowGuard=true;
      tbodyEl.addEventListener('click',function(e){
        if(!e.target.closest(
          'a.trace-link,a.rux-trace-link,'+
          '.query-clip,.query-toggle,.trace-detail-row,.trace-dropdown'
        )) e.stopPropagation();
      },true);
    }
    for(var i=0;i<bodyRows.length;i++){
      var row=bodyRows[i];
      // 仅 query 可点击展开详情：移除模板行级 onclick。
      row.removeAttribute('onclick');
      row.onclick=null;
      var cid=caseIdFromRow(row);
      var c=findCaseById(cases,cid);
      // 无「操作」列时，按模板固定的每页 10 条从当前筛选结果定位 case。
      if(!c&&filtered) c=filtered[pageOffset+i]||null;
      if(!cid&&c&&c.case_id!=null) cid=String(c.case_id);
      // output 列：display_output（输出的 result 文本），截断 40 字、可多行
      // 换行，悬浮显示全文（悬浮提示由 query-ux 的委托监听统一处理）
      if(oi>=0){
        if(insertedOutput&&row.children.length<thCount){
          var ntd=document.createElement('td');
          row.insertBefore(ntd, row.children[oi]||null);
        }
        var ocell=row.children[oi];
        if(ocell){
          var ofull=outputTextOf(c);
          ocell.textContent='';
          // 模板可能给 td 自带 data-full（其 setupCellHovers 会再叠一层
          // hover-tip）——移除，悬浮全文统一走 query-ux 的 query-tip-float。
          ocell.removeAttribute('data-full');
          if(!ofull){
            var om=document.createElement('span');
            om.style.color='#9ca3af';
            om.textContent='-';
            ocell.appendChild(om);
          }else{
            var oclip=document.createElement('span');
            oclip.className='output-clip';
            oclip.setAttribute('tabindex','0');
            oclip.setAttribute('data-full',ofull);
            var os=document.createElement('span');
            os.className='output-short';
            os.textContent=clipText(ofull,40);
            oclip.appendChild(os);
            ocell.appendChild(oclip);
          }
        }
      }
      // Trace链接列：模板只放 .trace-link-slot 占位，真实 href 来自 c.trace_url
      // （ieval_external-20260916：表无 #data-table 时本补丁早退，列恒为 '-'）。
      var ti=colIndex(table,'Trace链接');
      if(ti>=0){
        var tcell=row.children[ti];
        if(tcell&&!tcell.querySelector('a.trace-link,a.rux-trace-link')){
          var turl=(c&&c.trace_url)?String(c.trace_url).trim():'';
          tcell.textContent='';
          if(!turl){
            var tm=document.createElement('span');
            tm.style.color='#9ca3af';
            tm.textContent='-';
            tcell.appendChild(tm);
          }else{
            var ta=document.createElement('a');
            ta.href=turl;
            ta.target='_blank';
            ta.rel='noopener';
            ta.className='trace-link';
            ta.textContent='Trace ↗';
            tcell.appendChild(ta);
          }
        }
      }
    }
    // 综合分列右对齐加粗（数字列更美观，翻页重放幂等）
    var si=colIndex(table,'综合分');
    if(si>=0){
      var srows=table.querySelectorAll('tbody tr');
      for(var sr=0;sr<srows.length;sr++){
        var std=srows[sr].children[si];
        if(std){ std.style.textAlign='right'; std.style.fontWeight='600';
          std.style.fontVariantNumeric='tabular-nums'; }
      }
    }
    // 用例 ID 已从明细表移除，详情入口改由 query 下拉；去掉重复「操作」列。
    removeColByName(table,'操作');
  }

  // 详情模态框头部注入 Langfuse trace 外链（模板模态框无 trace 跳转入口；
  // 「点 traceID 看 trace 详情」链路：点击用例ID 弹页内模态框，框内可再
  // 跳 agentfuse trace 页。数据来自点击时记录的 window.__ruxActiveTraceUrl）。
  // 已存在 .rux-trace-link 时必须更新 href（否则二次打开仍指向首次 case
  // 的 stale Langfuse URL）。
  function injectTraceLink(){
    var url=String(window.__ruxActiveTraceUrl||'').trim();
    if(!url) return;
    var modal=document.querySelector('#detail-modal,.detail-modal');
    if(!modal) return;
    var existing=modal.querySelector('a.rux-trace-link');
    if(existing){ existing.href=url; return; }
    var head=modal.querySelector(
      '.detail-modal-header,#detail-modal-header,.detail-modal-title,#detail-modal-title,'+
      '.dm-header,.dm-title'
    )||modal;
    var a=document.createElement('a');
    a.className='rux-trace-link';
    a.href=url;
    a.target='_blank';
    a.rel='noopener';
    a.textContent='Trace ↗';
    head.appendChild(a);
  }

  // ── #6 「评测维度」单卡紧凑化：表格左 + 雷达居中；自绘雷达 + 图例 ──
  // 背景（ieval_external-20260916）：模板把表格/雷达拆成左右两卡，
  // 雷达图例常缺失或随雷达卡删除丢失；renderRadar 重绘不清空旧 canvas。
  function dimFullScore(cid,name){
    var id=String(cid||'').toLowerCase();
    return (id.indexOf('d0')===0||String(name||'').indexOf('安全')>=0)?1:2;
  }
  function hasYesterdayMeans(){
    var y=window.REPORT&&REPORT.yesterday_criterion_means;
    if(!y||typeof y!=='object') return false;
    for(var k in y){ if(Object.prototype.hasOwnProperty.call(y,k)) return true; }
    return false;
  }
  function compactDimTable(){
    var tblCard=document.getElementById('dim-table-card');
    var tbl=tblCard?tblCard.querySelector('table'):null;
    if(!tbl){
      var ths=document.querySelectorAll('th');
      for(var i=0;i<ths.length;i++){
        if((ths[i].textContent||'').trim()==='维度'){ tbl=ths[i].closest('table'); break; }
      }
    }
    if(!tbl||tbl.getAttribute('data-rux-dt')) return;
    var di=colIndex(tbl,'维度'), wi=colIndex(tbl,'权重');
    if(di<0) return;
    tbl.setAttribute('data-rux-dt','1');
    function mark(idx,cls){
      var ths=tbl.querySelectorAll('thead th');
      if(ths[idx]) ths[idx].classList.add(cls);
      var rows=tbl.querySelectorAll('tbody tr');
      for(var r=0;r<rows.length;r++){
        var td=rows[r].children[idx];
        if(td) td.classList.add(cls);
      }
    }
    mark(di,'rux-dim-name');
    if(wi>=0) mark(wi,'rux-dim-weight');
  }
  function ensureRadarLegend(pane){
    var legend=document.getElementById('radar-legend')||
      (pane&&pane.querySelector('.radar-legend'));
    if(legend&&legend.getAttribute('data-rux-leg')){
      if(pane&&legend.parentNode!==pane) pane.appendChild(legend);
      return legend;
    }
    if(!legend){
      legend=document.createElement('div');
      legend.id='radar-legend';
      legend.className='radar-legend';
    }
    legend.setAttribute('data-rux-leg','1');
    legend.innerHTML='';
    window.__ruxRadarOn=window.__ruxRadarOn||{today:true,yesterday:true};
    var yest=hasYesterdayMeans();
    if(!yest) window.__ruxRadarOn.yesterday=false;
    function addItem(key,label,swClass,locked){
      var it=document.createElement('div');
      it.className='rux-leg';
      it.setAttribute('data-series',key);
      var sw=document.createElement('span');
      sw.className='rux-sw '+swClass;
      var tx=document.createElement('span');
      tx.className='rux-leg-txt';
      tx.textContent=label;
      it.appendChild(sw); it.appendChild(tx);
      if(locked){ it.style.cursor='default'; }
      else{
        it.addEventListener('click',function(){
          window.__ruxRadarOn[key]=!window.__ruxRadarOn[key];
          it.classList.toggle('off',!window.__ruxRadarOn[key]);
          if(typeof window.renderRadar==='function'){
            try{ window.renderRadar(); }catch(e){ console.error('rux legend',e); }
          }
        });
      }
      it.classList.toggle('off',!window.__ruxRadarOn[key]);
      legend.appendChild(it);
    }
    addItem('today','今日','today',!yest);
    if(yest) addItem('yesterday','昨日','yesterday',false);
    if(pane) pane.appendChild(legend);
    return legend;
  }
  function installRadarRenderer(){
    if(window.renderRadar&&window.renderRadar.__ruxOwn) return;
    window.__ruxRadarOn=window.__ruxRadarOn||{today:true,yesterday:true};
    window.renderRadar=function(){
      var box=document.getElementById('radar-box')||
        document.getElementById('radar-chart-box');
      if(!box) return;
      var cs=box.querySelectorAll('canvas');
      for(var i=0;i<cs.length;i++){ if(cs[i].parentNode) cs[i].parentNode.removeChild(cs[i]); }
      var report=window.REPORT||{};
      var criteria=report.criteria||[];
      if(!criteria.length) return;
      var today=report.today_criterion_means||{};
      var yest=report.yesterday_criterion_means||{};
      var cm=report.criterion_metrics||{};
      function meanOf(map,cid){
        var v=map&&map[cid];
        if(typeof v==='number') return v;
        var s=cm[cid];
        if(s&&typeof s.mean==='number') return s.mean;
        return null;
      }
      var W=box.clientWidth||320, H=box.clientHeight||240;
      var dpr=window.devicePixelRatio||1;
      var canvas=document.createElement('canvas');
      canvas.width=Math.max(1,Math.round(W*dpr));
      canvas.height=Math.max(1,Math.round(H*dpr));
      canvas.style.width='100%'; canvas.style.height='100%';
      var tip=document.getElementById('radar-tooltip');
      if(!tip){
        tip=document.createElement('div');
        tip.id='radar-tooltip';
        tip.className='chart-tooltip';
        tip.style.cssText='position:absolute;display:none;pointer-events:none;'+
          'background:#1a1d2e;color:#fff;padding:4px 8px;border-radius:4px;'+
          'font-size:11px;z-index:5';
        box.appendChild(tip);
      }
      if(tip.parentNode!==box) box.appendChild(tip);
      box.style.position='relative';
      box.insertBefore(canvas, tip);
      var ctx=canvas.getContext('2d');
      if(!ctx) return;
      ctx.setTransform(dpr,0,0,dpr,0,0);
      var cx=W/2, cy=H/2, R=Math.max(28,Math.min(W,H)/2-36);
      var N=criteria.length;
      var pts=[];
      function xy(i,v){
        var ang=(-90+i*360/N)*Math.PI/180;
        return {x:cx+R*v*Math.cos(ang), y:cy+R*v*Math.sin(ang)};
      }
      ctx.strokeStyle='#e8e9ee'; ctx.lineWidth=1;
      var rings=[0.25,0.5,0.75,1];
      for(var r=0;r<rings.length;r++){
        ctx.beginPath();
        for(var i=0;i<=N;i++){
          var p=xy(i%N,rings[r]);
          if(i===0) ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y);
        }
        ctx.closePath(); ctx.stroke();
      }
      for(var i=0;i<N;i++){
        var p=xy(i,1);
        ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(p.x,p.y); ctx.stroke();
      }
      ctx.fillStyle='#6b7194';
      ctx.font="11px -apple-system,'SF Pro Text',sans-serif";
      for(var i=0;i<N;i++){
        var c=criteria[i]||{};
        var cid=String(c.id||'');
        var name=String(c.name||cid);
        var full=dimFullScore(cid,name);
        var short=name.length>8?name.slice(0,8)+'\u2026':name;
        var label=short+' /'+full;
        var lp=xy(i,1.18);
        ctx.textAlign=lp.x<cx?'right':(lp.x>cx?'left':'center');
        ctx.textBaseline='middle';
        ctx.fillText(label,lp.x,lp.y);
      }
      function drawSeries(map,color,dash,on){
        if(!on) return;
        ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.setLineDash(dash||[]);
        ctx.beginPath();
        for(var i=0;i<=N;i++){
          var c=criteria[i%N]||{};
          var cid=String(c.id||'');
          var full=dimFullScore(cid,c.name||cid);
          var m=meanOf(map,cid);
          var v=(m==null)?0:Math.max(0,Math.min(1,m/full));
          var p=xy(i%N,v);
          if(i===0) ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y);
          if(i<N){
            var rad=(m!=null&&m/full>=0.95)?5:3.5;
            pts.push({x:p.x,y:p.y,name:String(c.name||cid),mean:m,full:full,color:color,r:rad});
          }
        }
        ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
        for(var j=pts.length-N;j<pts.length;j++){
          if(j<0) continue;
          var q=pts[j];
          ctx.beginPath(); ctx.fillStyle=color;
          ctx.arc(q.x,q.y,q.r,0,Math.PI*2); ctx.fill();
          ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.stroke();
        }
      }
      var st=window.__ruxRadarOn||{today:true,yesterday:true};
      var todayMap=Object.keys(today).length?today:cm;
      drawSeries(todayMap,'#4f46e5',[],st.today!==false);
      if(hasYesterdayMeans()) drawSeries(yest,'#d97706',[8,4],st.yesterday!==false);
      canvas.onmousemove=function(ev){
        var rect=canvas.getBoundingClientRect();
        var mx=ev.clientX-rect.left, my=ev.clientY-rect.top;
        var hit=null, best=12;
        for(var i=0;i<pts.length;i++){
          var dx=pts[i].x-mx, dy=pts[i].y-my, d=Math.sqrt(dx*dx+dy*dy);
          if(d<best){ best=d; hit=pts[i]; }
        }
        if(!hit){ tip.style.display='none'; return; }
        var mv=hit.mean==null?'-':Number(hit.mean).toFixed(2);
        tip.textContent=hit.name+'：'+mv+' / '+hit.full;
        tip.style.display='block';
        tip.style.left=Math.min(mx+10,W-120)+'px';
        tip.style.top=Math.max(4,my-28)+'px';
      };
      canvas.onmouseleave=function(){ tip.style.display='none'; };
    };
    window.renderRadar.__ruxOwn=true;
    window.renderRadar.__ruxRR=true;
  }
  function compactDimSection(){
    compactDimTable();
    var tblCard=document.getElementById('dim-table-card');
    var radarCard=document.getElementById('radar-card');
    var box=document.getElementById('radar-box')||document.getElementById('radar-chart-box');
    var pane=null;
    var wrapper=null;
    if(tblCard){
      wrapper=tblCard.querySelector('.table-wrapper');
      if(!wrapper){
        var dimTbl=tblCard.querySelector('table');
        wrapper=dimTbl?dimTbl.parentNode:null;
      }
      if(wrapper&&!tblCard.contains(wrapper)) wrapper=null;
    }
    if(tblCard&&box&&wrapper&&!tblCard.getAttribute('data-rux-dim')){
      tblCard.setAttribute('data-rux-dim','1');
      var body=document.createElement('div');
      body.style.display='flex';
      body.style.alignItems='center';
      body.style.gap='12px';
      wrapper.style.flex='0 1 auto';
      wrapper.style.minWidth='0';
      wrapper.style.maxWidth='58%';
      if(wrapper.parentNode===tblCard) tblCard.insertBefore(body,wrapper);
      else tblCard.appendChild(body);
      body.appendChild(wrapper);
      pane=document.createElement('div');
      pane.style.flex='1 1 0';
      pane.style.minWidth='260px';
      pane.style.display='flex';
      pane.style.flexDirection='column';
      pane.style.alignItems='center';
      pane.style.justifyContent='center';
      var cap=document.createElement('div');
      cap.style.cssText='font-size:12px;color:#6b7194;margin-bottom:6px;text-align:center';
      cap.textContent='维度雷达（今日 vs 昨日）';
      pane.appendChild(cap);
      box.style.width='100%';
      box.style.maxWidth='320px';
      box.style.height='240px';
      pane.appendChild(box);
      var tip=document.getElementById('radar-tooltip');
      if(tip&&tip.parentNode!==box) box.appendChild(tip);
      var existingLeg=document.getElementById('radar-legend')||
        (radarCard&&radarCard.querySelector('.radar-legend'));
      if(existingLeg) pane.appendChild(existingLeg);
      ensureRadarLegend(pane);
      body.appendChild(pane);
      if(radarCard&&radarCard.parentNode){
        var row=radarCard.parentNode;
        radarCard.parentNode.removeChild(radarCard);
        if(row&&row.style) row.style.gridTemplateColumns='1fr';
      }
    }
    if(!pane&&box){
      pane=box.parentNode;
      ensureRadarLegend(pane);
    }
    installRadarRenderer();
    if(typeof window.renderRadar==='function'){
      try{ window.renderRadar(); }catch(e){ console.error('rux renderRadar',e); }
    }
  }

  // ── #4 综合归因 → 中文表格（分类|进度条|个数|百分比）────────────────
  function isAttrFieldTitle(el,field){
    if(!el) return false;
    var cls=el.classList;
    var named=cls&&(cls.contains('dist-field-title')||cls.contains('attr-field-title'));
    // ieval 模板字段标题常是无 class 的加粗 div
    var t=(el.textContent||'').trim();
    if(!t||t.length>80) return false;
    var tl=t.toLowerCase();
    var match=t.indexOf('主要问题类别')!==-1||
      tl.indexOf('primary_issue_category')!==-1||
      t.indexOf('扣分维度')!==-1||
      (field&&t.indexOf(field)!==-1);
    if(!match) return false;
    return named||el.tagName==='DIV';
  }
  function rebuildAttrTable(){
    var sec=document.getElementById('attribution-section');
    if(!sec) return;
    var am=window.REPORT&&REPORT.attribution_metrics;
    if(!am||typeof am!=='object') return;
    // fused（综合归因）：ieval=primary_issue_category；search_eval=扣分维度
    var entry=am['fused'];
    if(!entry) return;
    var fds=entry.field_distributions||{};
    var vls=entry.value_labels||{};
    var field=fusedAttrField();
    var dist=fds[field];
    if(!dist||typeof dist!=='object') return;
    var entries=Object.keys(dist).map(function(v){return {val:v,count:dist[v]};});
    entries.sort(function(a,b){return b.count-a.count;});
    var total=entries.reduce(function(s,e){return s+e.count;},0);
    var maxCount=entries.length?entries[0].count:1;
    // 定位「综合归因」块：search/ieval 模板用 .attr-title；旧模板偶发
    // .dist-section-title。rebuild 未命中则 search 的「扣分维度」停在
    // 无色 .dist-bar-fill 条上（宽有色无 → 进度条不可见）。
    var titles=sec.querySelectorAll('.attr-title,.dist-section-title');
    var fusedTitle=null;
    for(var i=0;i<titles.length;i++){
      if((titles[i].textContent||'').indexOf('综合归因')!==-1){ fusedTitle=titles[i]; break; }
    }
    if(!fusedTitle) return;
    // 字段标题：.attr-field-title / .dist-field-title / 无 class 加粗 div
    var fieldTitle=null, node=fusedTitle.nextElementSibling;
    while(node){
      if(isAttrFieldTitle(node,field)){ fieldTitle=node; break; }
      // 标题可能包在 wrapper 内：继续向下扫同级 + 一层子节点
      if(node.querySelector){
        var inner=node.querySelector('.attr-field-title,.dist-field-title');
        if(inner&&isAttrFieldTitle(inner,field)){ fieldTitle=inner; break; }
      }
      node=node.nextElementSibling;
    }
    if(!fieldTitle) return;
    // 收集 fieldTitle 之后连续的 dist-row / 旧表并移除（已有表格则先清）
    var host=fieldTitle.parentNode;
    var nn=fieldTitle.nextElementSibling;
    while(nn&&((nn.classList&&(nn.classList.contains('dist-row')||
      nn.classList.contains('attr-tbl')))||nn.tagName==='TABLE')){
      var nxt=nn.nextElementSibling;
      host.removeChild(nn);
      nn=nxt;
    }
    // 构建表格
    var tbl=document.createElement('table');
    tbl.className='attr-tbl';
    tbl.innerHTML='<thead><tr><th>分类</th><th>分布</th><th>个数</th><th>百分比</th></tr></thead>';
    var tbody=document.createElement('tbody');
    entries.forEach(function(e){
      var raw=e.val;
      var vlMap=vls[field]||{};
      var label=vlMap[raw]?vlMap[raw]:raw;
      // 扣分维度 key(中文名) → 展示中文名；ieval 走 CN_MAP
      if(field==='扣分维度'){
        var dm=String(raw).match(/^([^(]+)\\((.+)\\)$/);
        if(dm) label=dm[2];
      } else {
        label=cnLabel(label);
      }
      var barW=maxCount?(e.count/maxCount*100):0;
      var pct=total?(e.count/total*100):0;
      var tr=document.createElement('tr');
      tr.innerHTML='<td class="ac-cat">'+esc(label)+'</td>'+
        '<td class="ac-bar"><div class="ac-track">'+
        '<div class="ac-fill" style="width:'+barW.toFixed(1)+'%"></div></div></td>'+
        '<td class="ac-cnt">'+e.count+'</td>'+
        '<td class="ac-pct">'+pct.toFixed(1)+'%</td>';
      // 点击联动根因筛选（保留原 dist-row 行为；传原始分布键）
      (function(rv){ tr.style.cursor='pointer';
        tr.onclick=function(){ if(typeof filterByRootCause==='function') filterByRootCause(rv); };
      })(raw);
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    host.insertBefore(tbl, fieldTitle.nextSibling);
  }

  // ── #5 Case 明细默认 FAIL+POOR + 归因筛选口径对齐 ──────────────────
  // 归因统计把空/缺失 category 归一为 "unknown"；Case 明细仍可能是 ""。
  // 点击综合归因分类时切到「全部状态」，避免 FAIL+POOR 子集导致空表。
  // search_eval「扣分维度」：按 case 未满分维度匹配（无 primary_issue_category）。
  function normCat(v){
    if(v==null) return 'unknown';
    var s=String(v).trim();
    return s ? s : 'unknown';
  }
  function caseAttr(c){
    if(c&&c.attribution&&typeof c.attribution==='object') return c.attribution;
    var fused=c&&c.scores&&c.scores.fused;
    if(fused&&fused.attribution&&typeof fused.attribution==='object') return fused.attribution;
    return {};
  }
  function parseDeductedDim(rc){
    // "decision_completeness(决策信息完整度)" → {key, name}
    var m=String(rc||'').match(/^([^(]+)\\((.+)\\)$/);
    if(m) return {key:m[1].trim(), name:m[2].trim()};
    return {key:'', name:String(rc||'').trim()};
  }
  function dimFullScore(key, name){
    var k=String(key||'').toLowerCase();
    var n=String(name||'');
    if(k.indexOf('d0')===0||n.indexOf('安全')!==-1) return 1;
    return 2;
  }
  function readDimScore(c, key){
    if(!key||!c) return null;
    var scores=c.scores;
    if(scores&&typeof scores==='object'&&scores[key]&&typeof scores[key]==='object'){
      var s=Number(scores[key].score);
      if(Number.isFinite(s)) return s;
    }
    var sub=c.sub_scores;
    if(sub&&typeof sub==='object'&&sub[key]!=null){
      var raw=sub[key];
      var sv=Number(typeof raw==='object'&&raw?raw.score:raw);
      if(Number.isFinite(sv)) return sv;
    }
    // 直存形态：scores.<criterion>.sub_scores / judge.criterion_scores
    if(scores&&typeof scores==='object'){
      var cids=Object.keys(scores);
      for(var i=0;i<cids.length;i++){
        var cd=scores[cids[i]];
        if(!cd||typeof cd!=='object') continue;
        var ss=cd.sub_scores;
        if(ss&&typeof ss==='object'&&ss[key]!=null){
          var v=Number(ss[key]);
          if(Number.isFinite(v)) return v;
        }
        var judge=cd.judge;
        var cs=judge&&judge.criterion_scores;
        if(cs&&typeof cs==='object'){
          var jids=Object.keys(cs);
          for(var j=0;j<jids.length;j++){
            var payload=cs[jids[j]];
            var pss=payload&&payload.sub_scores;
            if(pss&&typeof pss==='object'&&pss[key]!=null){
              var pv=Number(pss[key]);
              if(Number.isFinite(pv)) return pv;
            }
          }
        }
      }
    }
    return null;
  }
  function matchDeductedDim(c, rc){
    var p=parseDeductedDim(rc);
    if(p.key){
      var sc=readDimScore(c, p.key);
      if(sc!=null) return sc < dimFullScore(p.key, p.name);
    }
    // 回退：root_cause_display 含中文名（多维用 / 连接）
    var display=String((c&&c.root_cause_display)||'').trim();
    if(!display) return false;
    var needle=p.name||String(rc||'').trim();
    if(!needle) return false;
    if(display===needle||display===String(rc)) return true;
    var parts=display.split('/');
    for(var i=0;i<parts.length;i++){
      if(parts[i]===needle) return true;
    }
    return false;
  }
  function matchRootCause(c, rc){
    if(!rc) return true;
    var field=fusedAttrField();
    if(field==='扣分维度') return matchDeductedDim(c, rc);
    // ieval 等：primary_issue_category（空 → unknown）
    return normCat(caseAttr(c).primary_issue_category) === normCat(rc);
  }
  // 状态下拉/搜索框 id 随模板漂移（case-grade-filter vs grade-filter，
  // search_eval-20260916 即后者）——防御归一查找，找不到返回 null 不阻断。
  function findGradeFilter(){
    var g=document.getElementById('case-grade-filter')||document.getElementById('grade-filter');
    if(g) return g;
    var sels=document.querySelectorAll('select');
    for(var i=0;i<sels.length;i++){
      var opts=sels[i].options,hit=0;
      for(var j=0;j<opts.length;j++){
        var ov=String(opts[j].value||'');
        if(ov==='FAIL'||ov==='POOR') hit++;
      }
      if(hit>=2) return sels[i];
    }
    return null;
  }
  function findCaseSearch(){
    return document.getElementById('case-search')||
      document.querySelector('.search-box')||
      document.querySelector('input[type="search"]');
  }
  function setupDefaultFilter(){
    var sel=findGradeFilter();
    // sel 缺失/已处理仅跳过「插入选项+默认选中」，不得早退——下方
    // applyFilters/filterByRootCause/renderTable 包装必须始终挂上
    // （search_eval-20260916：id 漂移早退导致综合归因点选空表、列整形丢失）。
    if(sel&&!sel.getAttribute('data-rux')){
      sel.setAttribute('data-rux','1');
      // 插入「不合格(FAIL+POOR)」选项并设为默认
      var hasFP=false;
      for(var i=0;i<sel.options.length;i++){
        if(sel.options[i].value==='FAIL_POOR'){hasFP=true;break;}
      }
      if(!hasFP){
        var opt=document.createElement('option');
        opt.value='FAIL_POOR';
        opt.textContent='不合格(FAIL+POOR)';
        // 插在「全部状态」之后
        if(sel.options.length&&sel.options[0].value===''){ sel.insertBefore(opt,sel.options[1]); }
        else { sel.insertBefore(opt,sel.firstChild); }
      }
      sel.value='FAIL_POOR';
    }
    // 包装 applyFilters：统一 grade + search + 归因筛选（unknown 归一 / 扣分维度）
    var orig=window.applyFilters;
    if(typeof orig==='function'&&!orig.__ruxFP){
      var w=function(){
        var g=findGradeFilter();
        var gv=g?g.value:'';
        var searchEl=findCaseSearch();
        var search=searchEl?String(searchEl.value||'').trim():'';
        var rc=window.activeRootCause||null;
        window.filteredCases=(window.CASES||[]).filter(function(c){
          if(gv==='FAIL_POOR'){
            if(c.grade!=='FAIL'&&c.grade!=='POOR') return false;
          } else if(gv && gv!=='ALL' && c.grade!==gv){
            return false;
          }
          if(search){
            var mid=(c.trace_id===search)||(c.case_id===search)||(c.display_case_id===search);
            var mq=(c.display_input===search);
            // 「复制可见截断文本 → 搜索」：…/... 结尾回退 display_input 前缀
            // 匹配（与 query-ux 搜索包装器对齐，避免本包装覆盖后能力回归）。
            if(!mid&&!mq&&/(?:…|\\.\\.\\.)$/.test(search)){
              var qp=search.replace(/(?:…|\\.)+$/,'');
              if(qp) mq=String(c.display_input||'').trim().indexOf(qp)===0;
            }
            if(!mid&&!mq) return false;
          }
          if(!matchRootCause(c, rc)) return false;
          return true;
        });
        window.currentPage=1;
        // 调用均已 try 化/包装化：renderTable 走包装链（内部容错），
        // renderPagination/renderRootCauseTag 为模板原版，抛错不得阻断 reshape。
        try{ if(typeof renderTable==='function') renderTable(); }catch(e){
          console.error('rux applyFilters renderTable',e); }
        try{ if(typeof renderPagination==='function') renderPagination(); }catch(e){
          console.error('rux applyFilters renderPagination',e); }
        try{ if(typeof renderRootCauseTag==='function') renderRootCauseTag(); }catch(e){
          console.error('rux applyFilters renderRootCauseTag',e); }
        reshapeCaseColumns();
      };
      w.__ruxFP=true;
      window.applyFilters=w;
    }
    // 点击综合归因：选中分类时切到「全部状态」（与归因全量口径对齐）
    // 若模板未定义 filterByRootCause，补一个最小实现（search_eval 常见）
    if(typeof window.filterByRootCause!=='function'){
      window.filterByRootCause=function(val){
        window.activeRootCause=(window.activeRootCause===val)?null:val;
        if(typeof window.applyFilters==='function') window.applyFilters();
      };
    }
    var origFRC=window.filterByRootCause;
    if(typeof origFRC==='function'&&!origFRC.__ruxAlign){
      var wfrc=function(val){
        var selecting=window.activeRootCause!==val;
        if(selecting){
          var g=findGradeFilter();
          // 「全部状态」value 随模板漂移（'' 或 'ALL'）：置空未命中选项时
          // 回退首选项，避免下拉框空白（selectedIndex=-1）。
          if(g){ g.value=''; if(g.selectedIndex<0) g.selectedIndex=0; }
        }
        origFRC.apply(this, arguments);
      };
      wfrc.__ruxAlign=true;
      window.filterByRootCause=wfrc;
    }
    // 包装 renderTable：用例ID 列 + 去掉 output（try 容错：模板原版抛错
    // 不得阻断 reshape——ieval_external-20260916 模板行绑定闭包 bug 抛
    // TypeError 穿透包装链，reshapeCaseColumns 被跳过 → 用例ID 无悬浮/点击）。
    var origRT=window.renderTable;
    if(typeof origRT==='function'&&!origRT.__ruxCols){
      var wrt=function(){
        try{ origRT.apply(this,arguments); }catch(e){ console.error('rux renderTable orig',e); }
        reshapeCaseColumns();
      };
      wrt.__ruxCols=true;
      window.renderTable=wrt;
    }
    // 包装 openDetail：弹出后注入 Langfuse trace 外链入口（模板模态框无
    // trace 跳转；query-ux 也可能已包装过 openDetail，链式叠加即可）。
    var origOD=window.openDetail;
    if(typeof origOD==='function'&&!origOD.__ruxTrace){
      var wod=function(){
        origOD.apply(this,arguments);
        // 同步注入一次，避免 setTimeout 前用户已点到 stale 链；再补一帧
        // 兜底模板异步改写 modal DOM。
        try{ injectTraceLink(); }catch(e){}
        setTimeout(injectTraceLink,0);
      };
      wod.__ruxTrace=true;
      window.openDetail=wod;
    }
  }

  // ── #7 趋势图残留 Chart.js 守卫清除（search_eval-20260916）──
  // 模板用原生 Canvas 2D 画趋势图，却残留 Chart.js 存在性检查
  // （typeof Chart==='undefined' → 「图表库未加载」return）；页面按新规
  // 不引入任何图表库 → 守卫恒真 → 近 7 天趋势整节空白。
  // 仅当 renderTrend 是原生 2D 实现（无 new Chart）且含残留守卫时，
  // 注入 Chart 桩让守卫通过并重绘一次。
  function fixTrendChartGuard(){
    var fn=window.renderTrend;
    if(typeof fn!=='function') return;
    var src='';
    try{ src=fn.toString(); }catch(e){ return; }
    if(src.indexOf("typeof Chart==='undefined'")===-1) return;
    if(src.indexOf('new Chart')!==-1) return;
    if(typeof window.Chart==='undefined'){
      window.Chart=function(){};
      window.Chart.__ruxStub=true;
    }
    try{ window.renderTrend(); }catch(e){ console.error('rux renderTrend',e); }
  }

  function runAll(){
    try{ setupDefaultFilter(); }catch(e){ console.error('rux setupDefaultFilter',e); }
    try{ rebuildAttrTable(); }catch(e){ console.error('rux rebuildAttrTable',e); }
    try{ reshapeCaseColumns(); }catch(e){ console.error('rux reshapeCaseColumns',e); }
    try{ compactDimSection(); }catch(e){ console.error('rux compactDimSection',e); }
    try{ fixTrendChartGuard(); }catch(e){ console.error('rux fixTrendChartGuard',e); }
    // 综合归因点击联动后会重新 renderAttribution → 重新构建表格
    var origRA=window.renderAttribution;
    if(typeof origRA==='function'&&!origRA.__ruxAttr){
      var wra=function(){ origRA.apply(this,arguments); setTimeout(rebuildAttrTable,0); };
      wra.__ruxAttr=true;
      window.renderAttribution=wra;
    }
    // 触发一次默认筛选
    if(typeof window.applyFilters==='function'){ try{ window.applyFilters(); }catch(e){} }
  }
  // 执行时机：模板常在 DOMContentLoaded 里才把 renderTable/applyFilters
  // 挂上 window（ieval_external-20260915：同步执行时 wrap 落空断链）。
  // 只在 loading 等 DCL；interactive（DCL 已过、load 未到）必须立刻跑，
  // 否则 addEventListener('DOMContentLoaded') 永远不会触发。
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',runAll);
  else runAll();
})();
</script>
"""

#: Report enhancement sections: Banner / score-loss waterfall / dimension period-over-period /
#: root-cause x dimension loss matrix. Restyle an existing section; build a missing one so nothing is duplicated.
#: Data comes from the new REPORT fields.
#: score_loss_waterfall / criterion_deltas / root_cause_dim_matrix.
#: Insert position: top of the container (Banner); before the "failed attribution" .sec (waterfall + delta + heatmap).
_REPORT_ENHANCE_PATCH = """
<style id="report-enhance-patch">
.qhub-banner{border-radius:12px;overflow:hidden;margin:0 0 16px;
background:linear-gradient(115deg,#0d3d2e 0%,#14532d 45%,#166534 100%);
color:#fff;padding:22px 28px;position:relative;box-shadow:0 4px 16px rgba(0,0,0,.08)}
.qhub-banner .qb-title{font-size:22px;font-weight:700;letter-spacing:.02em;margin:0 0 6px}
.qhub-banner .qb-sub{font-size:13px;opacity:.88;margin:0;padding-bottom:4px;
border-bottom:2px solid #2dd4bf;display:inline-block}
.enh-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:12px;margin:0 0 14px}
@media(max-width:900px){.enh-grid{grid-template-columns:1fr}}
.enh-card{background:#fff;border:1px solid #e2e4ea;border-radius:12px;padding:14px 16px;
box-shadow:0 1px 2px rgba(0,0,0,.03);position:relative}
.enh-card::before{content:"";position:absolute;left:0;right:0;top:0;height:3px;
border-radius:12px 12px 0 0;background:linear-gradient(90deg,#4f46e5,#818cf8)}
.enh-card h3{margin:0 0 10px;font-size:14px;font-weight:700;color:#1f2937}
.enh-wf{display:flex;align-items:stretch;gap:8px;height:190px;padding:4px 0 0}
.enh-wf-col{flex:1;display:flex;flex-direction:column;min-width:0}
.enh-wf-plot{flex:1;display:flex;flex-direction:column;justify-content:flex-end;
align-items:center;width:100%;min-height:0}
.enh-wf-bar{width:68%;border-radius:4px 4px 0 0;min-height:4px}
.enh-wf-bar.start{background:#4f46e5}.enh-wf-bar.loss{background:#f87171}
.enh-wf-bar.end{background:#16a34a}
.enh-wf-val{font-size:11px;font-weight:600;margin-bottom:4px;color:#374151}
.enh-wf-lab{height:28px;font-size:10px;color:#6b7194;margin-top:6px;text-align:center;
line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
overflow:hidden;word-break:break-all}
.enh-delta-list{list-style:none;margin:0;padding:0}
.enh-delta-list li{display:flex;justify-content:space-between;align-items:center;
padding:7px 0;border-bottom:1px solid #eef0f4;font-size:12.5px;gap:8px}
.enh-delta-list li:last-child{border-bottom:none}
.enh-delta-up{color:#16a34a}.enh-delta-down{color:#dc2626}.enh-delta-flat{color:#6b7194}
.enh-hm-wrap{overflow:auto;margin:0 0 14px}
.enh-hm{width:100%;border-collapse:collapse;font-size:12px}
.enh-hm th,.enh-hm td{border:1px solid #e5e7eb;padding:6px 5px;text-align:center}
.enh-hm th{background:#f8f9fb;color:#6b7194;font-weight:600;font-size:11px;white-space:nowrap}
.enh-hm td.rc{text-align:left;white-space:nowrap;font-weight:600;padding-left:8px}
.enh-note{font-size:11px;color:#6b7194;margin:8px 0 0}
.enh-sec{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#6b7194;
font-weight:700;margin:18px 0 10px}
#waterfall-delta-section{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);
column-gap:16px;align-items:start}
#waterfall-delta-section>.sec{grid-column:1/-1}
#waterfall-delta-section>.wd-grid{grid-column:1/-1;display:grid;
grid-template-columns:minmax(0,1fr) minmax(0,1fr);column-gap:16px}
@media(max-width:900px){#waterfall-delta-section{display:block}}
</style>
<script id="report-enhance-patch-js">
(function(){
  function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;')
    .replace(/</g,'&lt;').replace(/>/g,'&gt;');}
  function findSec(text){
    var secs=document.querySelectorAll('.sec');
    for(var i=0;i<secs.length;i++){
      if((secs[i].textContent||'').indexOf(text)!==-1) return secs[i];
    }
    return null;
  }
  function findMount(){
    return document.querySelector('.container') ||
      document.querySelector('main') ||
      document.body;
  }
  function normalizedText(value){
    return String(value||'').replace(/\\s+/g,'').toLowerCase();
  }
  function hasNativeSection(id,title){
    if(document.getElementById(id)) return true;
    var expected=normalizedText(title);
    var sections=document.querySelectorAll('.sec,.enh-sec');
    for(var i=0;i<sections.length;i++){
      if(normalizedText(sections[i].textContent)===expected) return true;
    }
    return false;
  }
  function injectBanner(mount){
    if(document.getElementById('qhub-banner')) return;
    var ban=document.createElement('div');
    ban.id='qhub-banner';
    ban.className='qhub-banner';
    ban.innerHTML='<p class="qb-title">AI Agent迭代质量</p>'+
      '<p class="qb-sub">评测结果速览，趋势一目了然</p>';
    var header=mount.querySelector('header');
    if(header&&header.parentNode===mount){ mount.insertBefore(ban, header); }
    else { mount.insertBefore(ban, mount.firstChild); }
  }
  function renderWaterfall(wf){
    if(!wf||!wf.steps||!wf.steps.length) return '';
    // 与满分同轴：柱高 = 值/start。禁止按 max(step.loss) 归一化——
    // 否则最大根因永远 height:100% 与满分齐平（ieval_external-20260915）。
    var start=Number(wf.start||2)||2;
    var end=Number(wf.end||0);
    var html='<div class="enh-wf">';
    html+='<div class="enh-wf-col"><div class="enh-wf-plot"><div class="enh-wf-val">'+
      start.toFixed(3)+'</div>'+
      '<div class="enh-wf-bar start" style="height:100%"></div></div>'+
      '<div class="enh-wf-lab">满分</div></div>';
    wf.steps.forEach(function(s){
      var loss=Number(s.loss)||0;
      var h=Math.max(4, Math.min(100, loss/start*100));
      html+='<div class="enh-wf-col"><div class="enh-wf-plot"><div class="enh-wf-val">-'+
        loss.toFixed(3)+'</div>'+
        '<div class="enh-wf-bar loss" style="height:'+h+'%"></div></div>'+
        '<div class="enh-wf-lab" title="'+esc(s.label)+'">'+esc(s.label)+'</div></div>';
    });
    html+='<div class="enh-wf-col"><div class="enh-wf-plot"><div class="enh-wf-val">'+
      end.toFixed(3)+'</div>'+
      '<div class="enh-wf-bar end" style="height:'+(Math.max(4, Math.min(100, end/start*100)))+
      '%"></div></div>'+
      '<div class="enh-wf-lab">综合分</div></div>';
    html+='</div><p class="enh-note">计算方法：总落差 = max(0, 满分 − 综合分)；'
      +'按各根因（含“未归因”）有效 case 的 fused 失分合计占比分摊，'
      +'各柱之和≈总落差。</p>';
    return html;
  }
  function renderDeltas(deltas){
    if(!deltas||!deltas.length) return '';
    var html='<ul class="enh-delta-list">';
    deltas.forEach(function(d){
      var delta=Number(d.delta)||0;
      var isFlat=Math.abs(delta)<=0.01;
      var cls=isFlat?'enh-delta-flat':(delta>0?'enh-delta-up':'enh-delta-down');
      var arrow=isFlat?'→':(delta>0?'↑':'↓');
      var sign=delta>=0?'+':'';
      html+='<li><span>'+esc(d.name||d.id)+'</span><span class="'+cls+'">'+arrow+' '+
        sign+delta.toFixed(3)+'（'+Number(d.today).toFixed(3)+' / '+Number(d.yesterday).toFixed(3)+
        '）</span></li>';
    });
    html+='</ul><p class="enh-note">计算方法：维度环比 = 今日同维度有效样本均分'
      +' − 昨日同 benchmark 已完成 run 的同维度有效样本均分；'
      +'绝对变化 ≤ 0.010 按持平显示，其余正值表示提升、负值表示下降。</p>';
    return html;
  }
  function renderHeatmap(mx){
    if(!mx||!mx.rows||!mx.cols||!mx.cells) return '';
    var maxCell=0;
    mx.rows.forEach(function(rc){
      var row=mx.cells[rc]||{};
      mx.cols.forEach(function(col){
        var v=Number(row[col.id]||0);
        if(v>maxCell) maxCell=v;
      });
    });
    if(!maxCell) maxCell=1;
    var html='<div class="enh-hm-wrap"><table class="enh-hm"><thead><tr><th>根因 × 维度</th>';
    mx.cols.forEach(function(col){
      var n=col.name||col.id;
      html+='<th title="'+esc(n)+'">'+esc(n.length>6?n.slice(0,6)+'…':n)+'</th>';
    });
    html+='</tr></thead><tbody>';
    mx.rows.forEach(function(rc){
      html+='<tr><td class="rc">'+esc(rc)+'</td>';
      var row=mx.cells[rc]||{};
      mx.cols.forEach(function(col){
        var raw=row[col.id];
        var hasValue=raw!==null&&raw!==undefined&&raw!=='';
        var v=hasValue?Number(raw):0;
        var a=v/maxCell;
        var bg='rgba(79,70,229,'+(0.08+a*0.75).toFixed(2)+')';
        var color=a>0.55?'#fff':(a>0.35?'#312e81':'#374151');
        html+='<td style="background:'+bg+';color:'+color+';font-weight:600">'+
          (hasValue&&Number.isFinite(v)?v.toFixed(3):'—')+'</td>';
      });
      html+='</tr>';
    });
    html+='</tbody></table></div>';
    html+='<p class="enh-note">计算方法：单元格 = 该根因（含“未归因”）下，'
      +'所有含该维有效分数 case 的（维度满分 − 得分）均值；'
      +'满分 case 按 0 计入，安全维满分 1，其余维度满分 2。</p>';
    return html;
  }
  function buildBlock(includeScoreLoss,includeMatrix){
    var R=window.REPORT||{};
    var wf=R.score_loss_waterfall;
    var deltas=R.criterion_deltas;
    var mx=R.root_cause_dim_matrix;
    var hasScoreLoss=includeScoreLoss&&(wf||(deltas&&deltas.length));
    var hasMatrix=includeMatrix&&mx;
    if(!hasScoreLoss&&!hasMatrix) return null;
    var wrap=document.createElement('div');
    wrap.id='report-enhance-block';
    var html='';
    if(hasScoreLoss){
      html+='<div class="enh-sec">分数损失与维度环比</div><div class="enh-grid">';
      if(wf){
        html+='<div class="enh-card"><h3>分数损失归因</h3>'+renderWaterfall(wf)+'</div>';
      }
      if(deltas&&deltas.length){
        html+='<div class="enh-card"><h3>维度环比（今日 vs 昨日）</h3>'
          +renderDeltas(deltas)+'</div>';
      }
      html+='</div>';
    }
    if(hasMatrix){
      html+='<div class="enh-sec">根因 × 维度失分矩阵</div>'+
        '<div class="enh-card">'+renderHeatmap(mx)+'</div>';
    }
    wrap.innerHTML=html;
    return wrap;
  }
  function injectEnhance(){
    try{
      var mount=findMount();
      if(!mount) return;
      injectBanner(mount);
      if(document.getElementById('report-enhance-block')) return;
      var hasScoreLoss=hasNativeSection(
        'waterfall-delta-section','分数损失与维度环比');
      var hasMatrix=hasNativeSection(
        'heatmap-section','根因 × 维度失分矩阵');
      // 原生瀑布 section 存在 → 整体替换为 patch 渲染：模板原生 .wf-*
      // 柱高塌缩不可靠（ieval_external-20260915：height 百分比在无确定
      // 高度的 flex 父级中塌缩为 min-height 细线，且柱底随标签折行漂移）。
      // 矩阵原生渲染正常，仅缺失时补建。
      var block=buildBlock(true,!hasMatrix);
      if(!block) return;
      if(hasScoreLoss){
        var nativeSec=document.getElementById('waterfall-delta-section');
        if(nativeSec&&nativeSec.parentNode){
          var secTitle=nativeSec.querySelector('.sec');
          var next=nativeSec.nextSibling;
          var parent=nativeSec.parentNode;
          parent.removeChild(nativeSec);
          parent.insertBefore(block,next);
          if(secTitle) block.insertBefore(secTitle,block.firstChild);
          return;
        }
      }
      var attrSec=findSec('不合格归因');
      if(attrSec&&attrSec.parentNode){
        attrSec.parentNode.insertBefore(block, attrSec);
      } else {
        var caseSec=findSec('Case 明细')||findSec('CASE 明细');
        if(caseSec&&caseSec.parentNode){
          caseSec.parentNode.insertBefore(block, caseSec);
        } else {
          mount.appendChild(block);
        }
      }
    }catch(e){ console.error('report-enhance', e); }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',injectEnhance);
  else setTimeout(injectEnhance,0);
})();
</script>
"""

#: Vertical-layout patch for the "per-dimension scores" column (injected before </body>, covering LLM template drift).
#:
#: Background: the prompt asks each criterion to occupy its own line, but LLM templates often wrap with a
#: newline plus white-space:pre-line. A global td rule (such as #data-table td{white-space:nowrap}) has
#: higher ID specificity than the cell class, so pre-line is overridden and dimensions sit on one line
#: (online regression in search_eval-20260906-b1).
#: The report showed dimensions squeezed onto a single row.
#:
#: This patch finds the column by the header text "per-dimension scores" (spaces/newlines in the header are ok),
#: sets inline white-space:pre-line on each td (inline style beats any stylesheet), and sets direct element
#: children (span and similar) to display:block. Both common paths (text wrap / inline spans side by side)
#: then stack vertically.
_DIM_SCORES_LAYOUT_PATCH = """
<script id="dim-scores-layout-patch">
(function(){
  function dimColIndex(table){
    var ths=table.querySelectorAll('thead th');
    for(var i=0;i<ths.length;i++){
      var t=String(ths[i].textContent||'').replace(/\\s+/g,'');
      if(t.indexOf('各维度得分')!==-1) return i;
    }
    return -1;
  }
  function layoutDimScores(){
    var tables=document.querySelectorAll('table');
    for(var ti=0;ti<tables.length;ti++){
      var ci=dimColIndex(tables[ti]);
      if(ci<0) continue;
      var rows=tables[ti].querySelectorAll('tbody tr');
      for(var r=0;r<rows.length;r++){
        var td=rows[r].children[ci];
        if(!td) continue;
        if(td.style.whiteSpace!=='nowrap') td.style.whiteSpace='nowrap';
        var kids=td.children;
        for(var k=0;k<kids.length;k++){
          if(kids[k].style.display!=='block') kids[k].style.display='block';
        }
      }
    }
  }
  function wrap(){
    var orig=window.renderTable;
    if(typeof orig==='function'&&!orig.__dimLayout){
      var wrapped=function(){orig.apply(this,arguments);layoutDimScores();};
      wrapped.__dimLayout=true;
      window.renderTable=wrapped;
    }
    layoutDimScores();
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',wrap);
  else wrap();
})();
</script>
"""

#: insertBefore reference-node fallback (injected before </body>, covering LLM template drift).
#:
#: LLM chart templates often make the tooltip a sibling of the mount container, or call insertBefore
#: before the node created by createElement is in the DOM. ``box.insertBefore(canvas, tooltip)`` throws
#: NotFoundError when the reference is not a child of box, the canvas never mounts, and the chart is blank
#: (same root cause as the blank radars in ieval_external-20260915 and search_eval-20260915. The HTML
#: structure fix _fix_chart_tooltip_nesting covers only some id variants). This patch wraps
#: Element.prototype.insertBefore at runtime: if the reference is not a child, degrade to append
#: (equivalent to appendChild). Normal calls keep their behavior, and any id-naming variant
#: is still covered.
_SAFE_INSERT_BEFORE_PATCH = """
<script id="safe-insert-before-patch">
(function(){
  if(Element.prototype.__amapSafeInsertBefore)return;
  var orig=Element.prototype.insertBefore;
  Element.prototype.insertBefore=function(newNode,refNode){
    if(refNode!=null&&refNode.parentNode!==this){return orig.call(this,newNode,null);}
    return orig.call(this,newNode,refNode);
  };
  Element.prototype.__amapSafeInsertBefore=true;
})();
</script>"""

#: Caller-injected patch for the "Trace link" column (injected before </body>, covering LLM template drift).
#:
#: The webdev-agent external-link red line is a semantic judgment: any prompt instruction to "create a
#: clickable external link" (a static <a> literal, a DOM API, or window.open) makes the LLM reject the
#: whole response (search_eval-20260914-q3: a fixed rejection phrase instead of HTML, refused by
#: _HTML_PREFIX_RE). The prompt therefore asks the LLM only for a plain-text slot column, and this patch
#: injects the real <a> per row from cases-data after the page is generated. The split matches
#: _fill_link_slots (the LLM never sees a link-building instruction).
#:
#: Row to case prefers the "case id" column text (try trace_id / display_case_id / case_id; a "first 3...last 3"
#: truncation is restored from the prefix and suffix). If the id is not filled yet, fall back to
#: filteredCases plus the current page. Table lookup must not require only #data-table: the LLM often
#: writes class=case-tbl / tbody#case-tbody (ieval_external-20260916 production page: missing #data-table
#: left the Trace link column as '-', while cases-data
#: still had a trace_url value).
#: Paging/filtering replay through the wrapped renderTable (same pattern as _QUERY_UX_PATCH) and are
#: idempotent. If the LLM did not generate the column, skip it and do not block.
_TRACE_LINK_PATCH = """
<script id="trace-link-patch">
(function(){
  function findCaseTable(){
    var table=document.getElementById('data-table');
    if(table) return table;
    var tbody=document.getElementById('case-tbody');
    if(tbody){
      table=tbody.parentElement;
      while(table&&table.tagName!=='TABLE') table=table.parentElement;
      if(table){ table.id='data-table'; return table; }
    }
    var cands=document.querySelectorAll(
      'table.case-tbl,#table-wrapper table,.table-wrapper table');
    var i,j,ths,t;
    for(i=0;i<cands.length;i++){
      ths=cands[i].querySelectorAll('thead th');
      for(j=0;j<ths.length;j++){
        t=String(ths[j].textContent||'').replace(/\\s+/g,'');
        if(t==='query'||t==='用例ID'){ cands[i].id='data-table'; return cands[i]; }
      }
    }
    ths=document.querySelectorAll('th');
    for(i=0;i<ths.length;i++){
      t=String(ths[i].textContent||'').replace(/\\s+/g,'');
      if(t!=='query'&&t!=='用例ID') continue;
      table=ths[i];
      while(table&&table.tagName!=='TABLE') table=table.parentElement;
      if(table){ table.id='data-table'; return table; }
    }
    return null;
  }
  function loadCases(){
    try{
      var el=document.getElementById('cases-data');
      return el?JSON.parse(el.textContent):[];
    }catch(e){return [];}
  }
  function colIndex(table,pattern){
    var ths=table.querySelectorAll('thead th');
    for(var i=0;i<ths.length;i++){
      var t=String(ths[i].textContent||'').replace(/\\s+/g,'');
      if(pattern.test(t)) return i;
    }
    return -1;
  }
  function findCase(cases,idText){
    if(!idText) return null;
    for(var i=0;i<cases.length;i++){
      var ids=[cases[i].trace_id,cases[i].display_case_id,cases[i].case_id];
      for(var j=0;j<ids.length;j++){
        if(ids[j]!=null&&String(ids[j]).trim()===idText) return cases[i];
      }
    }
    // 用例ID 列截断「前3...后3」时按前后缀还原（模板 renderTable 只写截断文本）
    var m=/^(.{3})\\.\\.\\.(.{3})$/.exec(idText);
    if(m){
      for(var k=0;k<cases.length;k++){
        var ids2=[cases[k].trace_id,cases[k].display_case_id,cases[k].case_id];
        for(var n=0;n<ids2.length;n++){
          var s=ids2[n]==null?'':String(ids2[n]).trim();
          if(s.length>6&&s.slice(0,3)===m[1]&&s.slice(-3)===m[2]) return cases[k];
        }
      }
    }
    return null;
  }
  function caseIdFromRow(row){
    // 优先用「用例ID」列文本匹配；否则从操作列「详情」按钮的
    // onclick="openDetail('{case_id}')" 提取 case_id 作为匹配锚点。
    var btn=row.querySelector('[onclick*="openDetail"]');
    if(!btn) return '';
    var m=String(btn.getAttribute('onclick')||'').match(/openDetail\\(\\s*['"]([^'"]+)['"]/);
    return m?m[1]:'';
  }
  function upgrade(){
    var table=findCaseTable();
    if(!table) return;
    var ti=colIndex(table,/Trace链接/i);
    if(ti<0) return;
    var ii=colIndex(table,/用例ID/);
    var cases=loadCases();
    if(!cases.length) return;
    var rows=table.querySelectorAll('tbody tr');
    var filtered=Array.isArray(window.filteredCases)?window.filteredCases:null;
    var page=Math.max(1,Number(window.currentPage)||1);
    var pageOffset=(page-1)*10;
    for(var r=0;r<rows.length;r++){
      var cells=rows[r].children;
      var td=cells[ti];
      if(!td||td.querySelector('a.trace-link')) continue;
      // 用例ID 列已截断显示：优先读 data-full 里的完整 traceID 匹配
      var idText='';
      if(ii>=0&&cells[ii]){
        var slot=cells[ii].querySelector('[data-full]');
        idText=slot?String(slot.getAttribute('data-full')||'').trim():
          String(cells[ii].textContent||'').trim();
      }
      if(!idText) idText=caseIdFromRow(rows[r]);
      var c=idText?findCase(cases,idText):null;
      if(!c&&filtered) c=filtered[pageOffset+r]||null;
      var url=c&&c.trace_url?String(c.trace_url):'';
      if(!url){
        if(!String(td.textContent||'').trim()) td.textContent='-';
        continue;
      }
      td.textContent='';
      var a=document.createElement('a');
      a.setAttribute('href',url);
      a.setAttribute('target','_blank');
      a.setAttribute('rel','noopener');
      a.className='trace-link';
      a.textContent='Trace ↗';
      td.appendChild(a);
    }
  }
  function wrap(){
    upgrade();
    var orig=window.renderTable;
    if(typeof orig==='function'&&!orig.__traceLink){
      var wrapped=function(){
        orig.apply(this,arguments);
        upgrade();
      };
      wrapped.__traceLink=true;
      window.renderTable=wrapped;
    }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',wrap);
  else wrap();
})();
</script>
"""

#: Pattern to locate </body> for injecting script tags when
#: the LLM-generated HTML lacks them (fallback path in _fix_case_data
#: and _inject_report_data).
_BODY_CLOSE_RE = re.compile(r"</body>", re.IGNORECASE)

#: After tbody.innerHTML the LLM template binds ``rows[idx].addEventListener`` with the outer loop's ``i``
#: (already page.length) and throws TypeError on undefined. That pierces every renderTable wrapper, so
#: case-id hover/click and the query/output enhancements are skipped (ieval_external-20260916). Change the
#: IIFE argument to the inner row loop's ``ri``.
_RENDER_TABLE_ROW_BIND_RE = re.compile(
    r"(rows\[idx\]\.addEventListener\(\s*'click'[\s\S]{0,400}?\}\)\()i(\);)"
)

#: Scoring-rule external-link slot. The generation platform forbids LLM output from embedding external
#: links (a violation rejects the whole response). The template asks the LLM to render the slot element
#: (plain-text style, no <a>). The real link is injected by the caller afterwards (_fill_link_slots).
#: The URL comes from the run-config snapshot ``rule_link_url`` (carried on the Rubric class; business-configurable).
_RULE_LINK_SLOT_PATTERN = re.compile(
    r'<span[^>]*class="[^"]*rule-link-slot[^"]*"[^>]*>.*?</span>',
    re.DOTALL,
)

#: Pattern to locate the <script type="application/json" id="report-data"> tag
#: in LLM-generated HTML. Used by _inject_report_data to replace the tag
#: content with the original ReportSpec (minus case_results) from the spec.
_REPORT_DATA_PATTERN = re.compile(
    r'<script\s+type="application/json"\s+id="report-data">\s*(.*?)\s*</script>',
    re.DOTALL,
)

#: Minimum required HTML structural tags. If any is missing, the LLM output
#: was truncated and the HTML must not be uploaded to OSS. ``</style>`` is
#: intentionally not required: valid templates may have no style block.
_REQUIRED_HTML_TAGS = ("<body", "</body>", "</html>")

#: Conditional data section: a non-empty spec field means the title must be rendered (missing = the LLM dropped the column).
_CONDITIONAL_SECTION_TITLES: tuple[tuple[str, str], ...] = (
    ("grade_distribution", "档位分布"),
    ("performance_metrics", "性能 & 效率"),
    ("trend_7d", "近 7 天趋势"),
    ("attribution_metrics", "不合格归因"),
)

#: Required data-section titles (must be rendered unconditionally).
_REQUIRED_SECTION_TITLES = ("效果指标", "评测维度")

#: Automatic insight threshold constants.
_INSIGHT_HIGH_COMPLETION_RATE = 90.0  # Completion rate above this counts as "almost everything finished".
_INSIGHT_LOW_SCORE_PCT = 85.0  # An overall-score percent below this counts as "the score is not high".
_INSIGHT_LOW_COMPLETION_RATE = 80.0  # A completion rate below this means the flow-through should be investigated.
_INSIGHT_TOP_ROOT_CAUSES = 2  # The insight compares the top N root causes by score loss.
_INSIGHT_MAX_CHARS = 200  # Cap on the rule-fallback banner body (including punctuation).
#: Context cap submitted to the insight model (report metrics + case-detail JSON).
_INSIGHT_PAYLOAD_MAX_CHARS = 120_000
_INSIGHT_CASE_TEXT_MAX = 280
_INSIGHT_TOP_PROBLEMS_MAX = 3
_INSIGHT_LLM_MIN_CHARS = 8
_INSIGHT_LLM_MAX_TOKENS = 2500
_INSIGHT_LLM_TEMPERATURE = 0.3
_INSIGHT_LLM_RETRIES = 2
_INSIGHT_LLM_RETRY_BASE_S = 0.5
_INSIGHT_SYSTEM_PROMPT = (
    "你是高德地图 Agent 评测报告分析专家。"
    "只输出一个合法 JSON 对象，不要 markdown 代码块，不要其它解释文字。"
)


@dataclass(frozen=True)
class _InsightChatRequest:
    """Inputs for one insight-model Chat call (the parameter object of ``_post_insight_chat``)."""

    base_url: str
    api_key: str
    model: str
    prompt: str
    timeout_s: float


_INSIGHT_USER_PROMPT = (
    "请基于下面的评测报告指标与 **完整 case 明细**，用中文生成「今日摘要」"
    "和「Top 问题深度剖析」。\n\n"
    "输出 JSON schema（字段名必须一致）：\n"
    "{\n"
    '  "summary": {\n'
    '    "conclusion": "一段总述：综合分、满足率、主要正向/负向驱动",\n'
    '    "bullets": ["要点1", "要点2", "要点3"],\n'
    '    "action": "关注项与可执行建议"\n'
    "  },\n"
    '  "top_problems": [\n'
    "    {\n"
    '      "rank": 1,\n'
    '      "name": "根因标签 · 中文名",\n'
    '      "trend": "趋势/集中特征一句话",\n'
    '      "affected_cases": 33,\n'
    '      "score_drag": "-0.36",\n'
    '      "share_pct": "38.4%",\n'
    '      "impacted_dims": "主要波及维度",\n'
    '      "chain": ["环节1", "环节2", "环节3", "结果"],\n'
    '      "typical_case_id": "用例ID",\n'
    '      "typical_case_desc": "典型案例一句话"\n'
    "    }\n"
    "  ],\n"
    '  "insight_text": "兼容旧横幅的一句话摘要（可与 conclusion 相同）"\n'
    "}\n\n"
    "硬性要求：\n"
    "1. 数字必须来自输入数据，禁止编造。"
    "score_drag / share_pct / affected_cases 优先从 metrics.score_loss_waterfall、"
    'attribution_metrics、case 统计推算；算不出则填空字符串 ""，'
    "**禁止**写「数据不足」「未知」「N/A」等占位文案（前端不展示空字段）。\n"
    "2. summary.bullets 2～4 条；top_problems 1～3 条，按影响优先排序。\n"
    "3. 文风参考：综合得分 X（±Δ），任务满足率…；根因驱动；POOR+FAIL 集中场景；"
    "优良率与连续天数；关注项建议。\n"
    "4. chain 为根因链路 3～5 步；typical_case_* 必须能在 case 明细中找到对应 ID。\n"
    '5. score_drag 须为可解析数值字符串（如 "-0.36"），不要带单位外的说明文字。\n'
    "6. 规则洞察仅作参考，以 case/指标数据为准。\n\n"
    "规则洞察：<<FALLBACK>>\n\n"
    "报告数据 JSON：\n<<PAYLOAD>>\n"
)
_INSIGHT_BANNER_RE = re.compile(
    r'(<(?:div|p|section)[^>]*class="[^"]*insight-banner[^"]*"[^>]*>)(.*?)'
    r"(</(?:div|p|section)>)",
    re.DOTALL | re.IGNORECASE,
)
_INSIGHT_PATCH_STYLE_RE = re.compile(
    r'<style\s+id="[^"]*(?:ux-patch|enhance|trace-link)[^"]*">.*?</style>',
    re.DOTALL | re.IGNORECASE,
)
_INSIGHT_PATCH_SCRIPT_RE = re.compile(
    r'<script\s+id="[^"]*(?:ux-patch|enhance|trace-link)[^"]*"[^>]*>.*?</script>',
    re.DOTALL | re.IGNORECASE,
)
_INSIGHT_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)
_INSIGHT_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_UNATTRIBUTED_ROOT_CAUSE_LABEL = "未归因"

#: English label to Chinese display map for primary_issue_layer / primary_issue_category.
#:
#: Used for the Chinese "root cause" column (aligned with the "overall attribution" stats table). value_labels
#: often map the raw category to ``"{layer}/{category}"`` (for example planning/strategy_error). After the
#: prefix is deduped, this table converts it to the display name. A layer that appears alone is also
#: converted (planning to its display name, and so on). Unknown labels stay as the original text.
_ROOT_CAUSE_CN_LABELS: dict[str, str] = {
    "planning/strategy_error": "规划/策略错误",
    "planning/intent_error": "规划/意图错误",
    "planning/context_lost": "规划/上下文丢失",
    "tool/tool_exec_error": "工具/执行错误",
    "tool/tool_unavailable": "工具/不可用",
    "tool/data_insufficient": "工具/数据不足",
    "generation/factuality_error": "生成/事实性错误",
    "generation/completeness_error": "生成/完整性错误",
    # layer appearing alone
    "planning": "规划",
    "tool": "工具",
    "generation": "生成",
    "none": "",
}


def _root_cause_cn(label: str) -> str:
    """Convert an English root-cause label to its display name (aligned with the overall-attribution table; unknown labels stay as-is)."""
    s = str(label or "").strip()
    if not s:
        return ""
    if s in _ROOT_CAUSE_CN_LABELS:
        return _ROOT_CAUSE_CN_LABELS[s]
    # An unknown combination such as planning/strategy_error: convert segment by segment.
    if "/" in s:
        parts = []
        for seg in s.split("/"):
            parts.append(_ROOT_CAUSE_CN_LABELS.get(seg, seg))
        return "/".join(parts)
    return s


def _numeric_score(value: object) -> float | None:
    """Coerce a score to a number: bool / non-numeric → ``None`` (excluded from the not-full-score decision)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _dimension_full_score(cid: str, name: str) -> float:
    """Per-dimension full score: a veto dimension whose id starts with d0/D0, or whose name contains the safety keyword, has full score 1; the rest are 2."""
    return 1.0 if cid.lower().startswith("d0") or "安全" in name else 2.0


def _score_full_mark(spec: dict[str, Any]) -> float:
    """Overall full score: the trace daily report passes ``score_full_mark`` through; otherwise the default is 2.0."""
    raw = spec.get("score_full_mark")
    if isinstance(raw, bool):
        return 2.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return 2.0
    return 2.0


#: Extract <style> block contents (used to detect the display:none anti-pattern).
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)

#: Extract CSS rules that hide elements (selector list, body). Only simple id/class selectors that can be
#: matched with certainty, so ``#section .tooltip`` is not mistaken for the whole section being hidden.
_CSS_HIDDEN_RE = re.compile(
    r"([^{}]+)\{[^{}]*(?:display\s*:\s*none|visibility\s*:\s*hidden)[^{}]*\}",
    re.IGNORECASE,
)

#: Pattern to find function names referenced in inline on* event attributes
#: (onclick, oninput, onchange, onkeydown, onkeyup, onsubmit).
#: Captures the function name before the opening parenthesis.
_ON_EVENT_FUNC_RE = re.compile(
    r'\bon(?:click|input|change|keydown|keyup|submit)\s*=\s*"([^"\s]*?)\s*\(',
)

#: Pattern to detect an IIFE (Immediately Invoked Function Expression) opening:
#: ``(function(){`` (with optional whitespace).
_IIFE_OPEN_RE = re.compile(r"\(\s*function\s*\(\s*\)\s*\{")

#: Pattern to find named function definitions: ``function funcName(``.
_FUNC_DEF_RE = re.compile(r"\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(")

#: Pattern to find ``<script>`` tags (any attributes) and capture attrs + body.
_SCRIPT_TAG_RE = re.compile(
    r"<script([^>]*)>(.*?)</script>",
    re.DOTALL | re.IGNORECASE,
)

#: Pattern to find the IIFE closing ``})();`` or ``})()`` (with optional
#: whitespace and trailing semicolon).
_IIFE_CLOSE_RE = re.compile(r"\}\s*\)\s*\(\s*\)\s*;?")

#: Pattern to find ``window.X = function(params) {`` — captures the
#: function name (group 1) and parameter list (group 2).
#: Used by _fix_window_function_hoisting to detect functions that
#: the LLM assigned to ``window`` inside an IIFE, which are only
#: available *after* the assignment executes (not hoisted).
_WINDOW_FUNC_ASSIGN_RE = re.compile(
    r"window\.(\w+)\s*=\s*function\s*\(([^)]*)\)\s*\{",
)

#: Pattern to find bare function calls ``NAME(`` — not preceded by ``.``
#: (method call) or an identifier char (name continuation).
_BARE_CALL_RE = re.compile(r"(?<![.\w$])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(")

#: Pattern to find function definitions via variable assignment:
#: ``const/let/var NAME = function`` / ``= (`` / ``= async (`` / ``= x =>``.
_VAR_FUNC_DEF_RE = re.compile(
    r"\b(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*"
    r"(?:async\s+)?(?:function\b|\(|[a-zA-Z_$][a-zA-Z0-9_$]*\s*=>)",
)

#: Pattern to strip JS string literals (single/double/backtick) before
#: Scanning for bare calls. String contents (such as CSS 'translate(-50%,-100%)') must not be
#: mistaken for a function call. Template strings are stripped whole (${} expressions are ignored too;
#: the conservative direction is to inject less, not to inject by mistake).
_JS_STRING_RE = re.compile(
    r"'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"|`(?:\\.|[^`\\])*`",
    re.DOTALL,
)

#: Pattern to strip JS comments (// line and /* */ block) before scanning —
#: Comment text (such as '// Draw fused_avg line (solid)') must not be mistaken for a function call.
#: This must run after string stripping (a string may contain //, such as a URL).
_JS_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)

#: JS keywords and common builtins — bare calls of these names are never
#: "called but undefined" problems and must be excluded from detection.
_JS_CALL_IGNORE = frozenset(
    {
        # keywords / control flow
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "function",
        "return",
        "new",
        "typeof",
        "void",
        "delete",
        "in",
        "of",
        "do",
        "else",
        "try",
        "finally",
        "with",
        "case",
        "throw",
        "instanceof",
        "await",
        "yield",
        "super",
        "async",
        # common builtins and global objects
        "setTimeout",
        "setInterval",
        "clearTimeout",
        "clearInterval",
        "requestAnimationFrame",
        "cancelAnimationFrame",
        "queueMicrotask",
        "Promise",
        "Array",
        "Object",
        "JSON",
        "Math",
        "Number",
        "String",
        "Boolean",
        "Date",
        "RegExp",
        "Error",
        "TypeError",
        "RangeError",
        "SyntaxError",
        "ReferenceError",
        "Map",
        "Set",
        "WeakMap",
        "WeakSet",
        "Symbol",
        "BigInt",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "encodeURIComponent",
        "decodeURIComponent",
        "encodeURI",
        "decodeURI",
        "escape",
        "unescape",
        "eval",
        "alert",
        "confirm",
        "prompt",
        "console",
        "log",
        "require",
        "define",
        "exports",
        "module",
        "document",
        "window",
        "navigator",
        "localStorage",
        "sessionStorage",
        "fetch",
        "XMLHttpRequest",
        "Intl",
        "Proxy",
        "Reflect",
        "structuredClone",
        "btoa",
        "atob",
        "Blob",
        "FileReader",
        "FormData",
        "URL",
        "URLSearchParams",
        "TextEncoder",
        "TextDecoder",
        "Image",
        "Audio",
        "Option",
        "Worker",
        "ResizeObserver",
        "MutationObserver",
        "IntersectionObserver",
    }
)

#: Verb-prefix table (render-function naming habit): "called renderXxx but defined drawXxx".
#: That kind of rename drift is recognized as an alias when the suffix matches after the prefix is stripped.
_FUNC_VERB_PREFIXES = (
    "render",
    "draw",
    "show",
    "update",
    "build",
    "display",
    "paint",
    "plot",
    "init",
    "setup",
    "load",
    "refresh",
    "create",
    "make",
    "generate",
)


def _strip_verb_prefix(name: str) -> str | None:
    """Strip the verb prefix of a function name (at the camelCase boundary) and return the suffix; return None if it cannot be stripped."""
    for prefix in _FUNC_VERB_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            rest = name[len(prefix) :]
            if rest[0].isupper():
                return rest
    return None


def _unique_suffix_alias(name: str, pool: set[str]) -> str | None:
    """Find the unique alias candidate in the defined-function pool that shares ``name``'s suffix (for example renderX→drawX).

    Two match levels, both requiring a unique candidate (on ambiguity return None and
    degrade to a stub, so nothing is bound by mistake):
    1. Exact suffix equality (renderTrendChart→drawTrendChart);
    2. Prefix containment — the LLM may write the call name one step shorter or longer:
       the candidate suffix starts with the call suffix (renderTrend→drawTrendChart, the
       missing trend-section regression in search_eval-20260910), or the call suffix
       starts with the candidate suffix (renderTrendChart→drawTrend).
    """
    suffix = _strip_verb_prefix(name)
    if not suffix:
        return None
    exact = sorted(other for other in pool if other != name and _strip_verb_prefix(other) == suffix)
    if exact:
        return exact[0] if len(exact) == 1 else None
    fuzzy = sorted(
        other
        for other in pool
        if other != name
        and (other_suffix := _strip_verb_prefix(other)) is not None
        and (other_suffix.startswith(suffix) or suffix.startswith(other_suffix))
    )
    return fuzzy[0] if len(fuzzy) == 1 else None


@dataclass
class _HtmlNode:
    """HTML node summary (only for report validation; not a full browser DOM)."""

    tag: str
    attrs: dict[str, str]
    text_parts: list[str]

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


class _ReportHtmlParser(HTMLParser):
    """Extract real node text, ignoring template strings and script text inside script/style."""

    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.nodes: list[_HtmlNode] = []
        self.visible_text_parts: list[str] = []
        self._stack: list[_HtmlNode] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if tag_l in {"script", "style"}:
            self._skip_depth += 1
        node = _HtmlNode(
            tag=tag_l,
            attrs={k.lower(): v or "" for k, v in attrs},
            text_parts=[],
        )
        self.nodes.append(node)
        if tag_l not in self._VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx].tag == tag_l:
                del self._stack[idx:]
                break

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        self.visible_text_parts.append(data)
        # Text belongs to the current node and its ancestors, so a parent section's hidden-check can see the title.
        for node in self._stack:
            node.text_parts.append(data)

    @property
    def visible_text(self) -> str:
        return "".join(self.visible_text_parts)


@dataclass(frozen=True)
class _SendSpecSummary:
    """Log payload for ``_log_send_spec_summary`` (FUNC-PARAMS-001 parameter consolidation)."""

    spec: dict[str, Any]
    prompt: str
    payload: dict[str, Any]


class WebdevReportClient:
    """A2A client for webdev-agent — generates HTML benchmark reports.

    Lifecycle:
        1. Construct A2A ``message/send`` JSON-RPC payload with ReportSpec.
        2. POST to ``webdev_agent_url``.
        3. Extract HTML from ``result.artifacts[0].parts[0].text``.
        4. Optionally upload to storage → ``sign_url`` (30d) → ``report_url``.

    The storage client (if provided) must satisfy the
    ``ReportStorageClient`` protocol (``upload(content_type=...)`` /
    ``sign_url``; Normandy / env credentials are handled inside that client).
    """

    def __init__(
        self,
        *,
        webdev_agent_url: str,
        timeout: float = _DEFAULT_TIMEOUT_S,
        storage_client: "ReportStorageClient | None" = None,
        insight_llm_complete: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self._url = webdev_agent_url.rstrip("/")
        self._timeout = timeout
        self._storage = storage_client
        self._insight_llm_complete = insight_llm_complete

    async def _try_unified_static_report(
        self,
        spec: dict[str, Any],
    ) -> tuple[str, str | None] | None:
        """Generate the report from the static unified template; return ``None`` when unavailable so the caller falls back to the LLM skill."""
        from benchmark.reporting.framework.unified_report.pipeline import (  # noqa: PLC0415
            generate_unified_report_from_spec,
            unified_report_enabled,
        )
        from benchmark.reporting.framework.unified_report.template_loader import (  # noqa: PLC0415
            UnifiedTemplateError,
        )

        if not unified_report_enabled():
            return None
        try:
            html_content = await generate_unified_report_from_spec(spec)
        except UnifiedTemplateError as exc:
            logger.warning(
                "webdev_client.unified_template_unavailable fallback=webdev_skill err=%s",
                exc,
            )
            return None
        if not html_content:
            return "", None
        html_content = await self._refine_insight_from_html(html_content, spec)
        report_url: str | None = None
        if self._storage is not None:
            try:
                report_url = await self._upload_to_storage(spec, html_content)
                logger.info(
                    "webdev_client.unified_storage_uploaded report_url=%s",
                    report_url,
                )
            except Exception:
                logger.warning(
                    "webdev_client.unified_storage_upload_failed",
                    exc_info=True,
                )
        return html_content, report_url

    async def generate_report(
        self,
        spec: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Send ReportSpec to webdev-agent; return ``(html, report_url | None)``.

        Args:
            spec: Report specification dict (benchmark name, scores,
                criteria, attribution distribution, etc.).

        Returns:
            ``(html_content, report_url)`` — ``report_url`` is ``None`` if
            the storage upload was skipped or failed. ``html_content`` is
            ``""`` if the LLM output was unusable (empty / non-HTML /
            truncated / JS syntax broken) after ``_GENERATION_MAX_ATTEMPTS``
            attempts — such HTML must never reach OSS.

        When static unified template is available (default), fill
        ``benchmark_report.html`` via ReportPayload plugins. Webdev-agent
        skill generation is used only when:

        - ``AGENT_EVAL_FORCE_WEBDEV_REPORT=1``, or
        - legacy ``AGENT_EVAL_UNIFIED_REPORT=0``, or
        - static template load fails (``UnifiedTemplateError``).
        """
        unified = await self._try_unified_static_report(spec)
        if unified is not None:
            return unified

        # Shared precomputation entry for the static-template and LLM paths (order and dependencies: enrich_report_spec).
        self.enrich_report_spec(spec)
        # Required Case-detail columns (query / case status / overall score / per-dimension scores /
        # Trace link) completeness check: a miss only warns and does not block. Missing data cannot be fixed by
        # regenerating HTML, and blocking would mean the report is never sent.
        case_field_problems = self._validate_case_required_fields(spec)
        if case_field_problems:
            logger.warning(
                "webdev_client.case_required_fields_missing "
                "benchmark_name=%s run_id=%s total=%d samples=%s",
                spec.get("benchmark_name"),
                spec.get("run_id"),
                len(case_field_problems),
                case_field_problems[:10],
            )

        prompt = self._build_template_prompt(spec)

        # Retry the whole generation when LLM output is unusable (empty / not HTML / truncated structure / JS syntax error).
        # A template flaw (for example an extra `}` at the end of the main render JS in search_eval-20260911) is a
        # one-off random event, and regenerating usually recovers. Network/HTTP errors are not retried here.
        # They propagate, and the caller report_spec.generate_benchmark_report
        # (REPORT_MAX_ATTEMPTS) covers them with backoff retries.
        html_content = ""
        for attempt in range(1, _GENERATION_MAX_ATTEMPTS + 1):
            html_content = await self._generate_html_once(spec, prompt, attempt=attempt)
            if html_content:
                break
            if attempt < _GENERATION_MAX_ATTEMPTS:
                logger.warning(
                    "webdev_client.generation_retry attempt=%d max_attempts=%d "
                    "— LLM output unusable, regenerating",
                    attempt,
                    _GENERATION_MAX_ATTEMPTS,
                )
        if not html_content:
            return "", None

        # After HTML post-processing, submit the report to the Amap LLM (gpt-5.6-sol) to rewrite the insight.
        # On failure keep the rule-based insight and do not block the upload.
        html_content = await self._refine_insight_from_html(html_content, spec)

        # Optional storage upload
        report_url: str | None = None
        if self._storage is not None:
            try:
                report_url = await self._upload_to_storage(spec, html_content)
                logger.info(
                    "webdev_client.storage_uploaded report_url=%s",
                    report_url,
                )
            except Exception:
                logger.warning(
                    "webdev_client.storage_upload_failed",
                    exc_info=True,
                )

        return html_content, report_url

    def _reject_unusable_html(self, html_content: str, request_id: str) -> bool:
        """HTML usability check (reject = True).

        A non-HTML prefix or a truncated structure (missing </body> and similar) is
        rejected (generate_report decides whether to retry). A missing structural element
        only warns (the LLM drifting from SKILL.md does not block). Extracted from
        ``_generate_html_once`` to keep the statement count down (PLR0915).
        """
        # Reject non-HTML responses: the LLM may return an error message or
        # truncated text instead of a valid HTML document.
        if not _HTML_PREFIX_RE.match(html_content):
            prefix = html_content[:80].replace("\n", "\\n")
            if _EXTERNAL_LINK_REFUSAL_RE.search(html_content):
                logger.warning(
                    "webdev_client.external_link_refused request_id=%s url=%s "
                    "prefix=%s length=%d — webdev-agent 外链红线整单拒绝",
                    request_id,
                    self._url,
                    prefix,
                    len(html_content),
                )
            else:
                logger.warning(
                    "webdev_client.not_html_response request_id=%s url=%s prefix=%s length=%d",
                    request_id,
                    self._url,
                    prefix,
                    len(html_content),
                )
            return True

        # Warn if HTML is suspiciously short — likely a truncated or minimal
        # response from the LLM (e.g. only a title, no report body).
        if len(html_content) < _MIN_REPORT_HTML_LEN:
            logger.warning(
                "webdev_client.short_html request_id=%s html_len=%d "
                "— webdev-agent may have returned a truncated response",
                request_id,
                len(html_content),
            )

        # Validate HTML structural completeness before any data injection.
        # If the LLM output was truncated (e.g. only CSS, no <body>), uploading
        # it to OSS would serve a blank page to end users.
        missing_tags = self._validate_html_completeness(html_content)
        if missing_tags:
            logger.error(
                "webdev_client.incomplete_html request_id=%s html_len=%d "
                "missing_tags=%s — rejecting upload (LLM output truncated)",
                request_id,
                len(html_content),
                missing_tags,
            )
            return True

        return False

    async def _generate_html_once(
        self,
        spec: dict[str, Any],
        prompt: str,
        *,
        attempt: int,
    ) -> str:
        """Request webdev-agent once and run every check and post-process; return ``""`` when unusable.

        Failure semantics: unusable LLM output (empty / not HTML / truncated / JS syntax
        error) returns ``""`` (generate_report decides whether to retry). Network / HTTP
        exceptions propagate (retried at the runner layer).
        """
        request_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": message_id,
                    "parts": [
                        {"kind": "text", "text": prompt},
                    ],
                },
            },
        }

        # --- Detailed log: full spec summary sent to webdev-agent ---
        self._log_send_spec_summary(
            request_id,
            _SendSpecSummary(spec=spec, prompt=prompt, payload=payload),
        )

        async with AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._url}/",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            logger.info(
                "webdev_client.http_response request_id=%s status=%d response_bytes=%d",
                request_id,
                resp.status_code,
                len(resp.content),
            )
            resp.raise_for_status()
            data = resp.json()

        # --- Detailed log: response structure ---
        self._log_response_structure(request_id, data)

        html_content = self._extract_html(data)

        # Log extracted HTML diagnostics: length, prefix, cases-data tag presence
        has_cases_tag = bool(_CASES_DATA_PATTERN.search(html_content)) if html_content else False
        logger.info(
            "webdev_client.extracted_html request_id=%s "
            "html_len=%d has_cases_data_tag=%s prefix=%s",
            request_id,
            len(html_content),
            has_cases_tag,
            html_content[:120].replace("\n", "\\n") if html_content else "(empty)",
        )

        if not html_content:
            logger.warning(
                "webdev_client.empty_html request_id=%s url=%s",
                request_id,
                self._url,
            )
            return ""

        if self._reject_unusable_html(html_content, request_id):
            return ""

        # Inject full ReportSpec data (minus case_results) as a JSON script
        # tag. The LLM generates only the template + JS rendering logic;
        # agent-eval injects the real data afterward.
        html_content = self._inject_report_data(html_content, spec)
        fixed_has_data = bool(_REPORT_DATA_PATTERN.search(html_content))
        logger.info(
            "webdev_client.after_inject_report_data request_id=%s "
            "html_len=%d has_report_data_tag=%s",
            request_id,
            len(html_content),
            fixed_has_data,
        )

        # Re-inject original case_results (PG source of truth) into HTML,
        # unconditionally replacing whatever the LLM wrote.
        html_content = self._fix_case_data(html_content, spec)
        fixed_has_cases = bool(_CASES_DATA_PATTERN.search(html_content))
        logger.info(
            "webdev_client.after_fix_case_data request_id=%s html_len=%d has_cases_data_tag=%s",
            request_id,
            len(html_content),
            fixed_has_cases,
        )

        # Fill external-link slots (scoring rules and similar). The platform forbids the LLM from emitting external links;
        # the real link is injected by this caller afterwards (URL from the run-config snapshot).
        html_content = self._fill_link_slots(html_content, spec.get("rule_link_url"))

        # Ensure <meta charset="UTF-8"> is present so browsers correctly
        # render Chinese text even when OSS does not set Content-Type charset.
        html_content = self._ensure_charset(html_content)

        # Hoist window.X = function() {} assignments to function declarations
        # so they are available throughout the IIFE scope (prevents
        # TypeError when called before the assignment executes).
        html_content = self._fix_window_function_hoisting(html_content)

        # Fix IIFE-scoped functions that inline onclick/oninput/onchange
        # handlers cannot access — causes ReferenceError on button clicks.
        html_content = self._fix_iife_global_scope(html_content)

        # Fallback for "called but not defined" functions: alias alignment / stub isolation, so init does not stop the whole page.
        html_content = self._fix_undefined_function_calls(html_content)

        # Fix the LLM template using the outer loop variable i in rows[idx].addEventListener.
        html_content = self._fix_render_table_row_bind(html_content)

        # Query truncation + hover with copyable full text + exact search on case id/query (covers LLM templates).
        html_content = self._inject_query_ux(html_content)

        # Case-detail column reshape (case id / drop output and actions) + overall-attribution table + default FAIL/POOR.
        html_content = self._inject_report_ux(html_content)

        # Enhancement sections: QHub Banner / score-loss waterfall / dimension delta / root-cause x dimension heatmap (add only).
        html_content = self._inject_report_enhance(html_content)

        # Fallback that stacks the "per-dimension scores" column vertically (covers LLM template drift).
        html_content = self._inject_dim_scores_layout(html_content)

        # Caller injection of the real link in the "Trace link" column (the LLM emits only a plain-text slot;
        # link-building stays out of the prompt: the external-link red line rejects the whole response).
        html_content = self._inject_trace_link_patch(html_content)

        # Runtime fallback for the insertBefore reference node: LLM chart templates often make the tooltip a
        # sibling of the mount container, or call insertBefore before createElement's node is in the DOM.
        # A reference that is not a child throws NotFoundError, the canvas never mounts, and the chart is blank
        # (same root cause as the blank radars in ieval_external-20260915 / search_eval-20260915).
        # A runtime wrap is more general than the HTML structure fix (_fix_chart_tooltip_nesting):
        # it covers any id-naming variant.
        html_content = self._inject_safe_insert_before_patch(html_content)

        # Chart tooltip structure fallback: the LLM may write the tooltip div as a sibling of chart-box.
        # JS insertBefore(canvas, tooltip) then throws NotFoundError and the canvas never mounts, so the chart
        # is blank (blank-radar incident in ieval_external-20260915).
        html_content = self._fix_chart_tooltip_nesting(html_content)

        # Isolate top-level init render steps with try/catch: one step failing must not stop later
        # section rendering (in the same incident renderRadar threw, so renderAttribution never ran and the
        # failed-attribution block was blank — the prompt asks for isolation, but the LLM often ignores it).
        html_content = self._isolate_init_steps(html_content)

        # JS syntax fallback check (after all post-processing): a syntax flaw in the LLM render script fails the
        # whole block at browser parse time, so no render function runs (data is injected but the page is an
        # empty template). A runtime try/catch cannot catch a parse-time SyntaxError, so it must be caught
        # before the OSS upload and trigger regeneration.
        js_problems = self._validate_js_scripts(html_content)
        if js_problems:
            logger.warning(
                "webdev_client.js_syntax_invalid request_id=%s attempt=%d "
                "problems=%s — rejecting upload (LLM template JS syntax broken)",
                request_id,
                attempt,
                js_problems,
            )
            return ""

        # Data-domain completeness and visibility check (last gate): data tags are non-empty and parseable, and
        # data sections exist and are not hidden by a stylesheet (ieval_external-20260914 online incident:
        # stylesheet ``display:none`` plus JS ``style.display=''`` clearing the inline style does not override
        # it, so tier distribution / trend / attribution / CASE details stayed blank — data injection was fine).
        data_problems = self._validate_data_sections(html_content, spec)
        if data_problems:
            logger.warning(
                "webdev_client.data_sections_invalid request_id=%s attempt=%d "
                "problems=%s — rejecting upload (data sections incomplete/hidden)",
                request_id,
                attempt,
                data_problems,
            )
            return ""

        return html_content

    @staticmethod
    def _ensure_charset(html: str) -> str:
        """Ensure the HTML has a ``<meta charset="UTF-8">`` declaration.

        If the LLM-generated HTML lacks a charset meta tag, inject one
        right after ``<head>`` (or ``<html>`` if no ``<head>`` is present).
        This prevents browser mojibake when OSS serves the file with
        ``Content-Type: text/html`` (no charset).
        """
        if _CHARSET_RE.search(html):
            return html

        meta = '<meta charset="UTF-8">'

        m = _HEAD_OPEN_RE.search(html)
        if m:
            return html[: m.end()] + meta + html[m.end() :]

        m = _HTML_OPEN_RE.search(html)
        if m:
            return html[: m.end()] + meta + html[m.end() :]

        return meta + html

    @staticmethod
    def _inject_query_ux(html: str) -> str:
        """Inject query truncation + exact-search patch before ``</body>``.

        LLM templates vary; this patch wraps ``applyFilters`` / ``renderTable``
        / ``openDetail`` so Case-detail search is an exact match on case id and query, the query
        cell and the detail modal are shown truncated, and the hover layer can copy the full text.
        """
        if 'id="query-ux-patch"' in html:
            return html
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            return html[: body_close.start()] + _QUERY_UX_PATCH + html[body_close.start() :]
        return html + _QUERY_UX_PATCH

    @staticmethod
    def _inject_report_ux(html: str) -> str:
        """Inject the Case-detail column reshape + overall-attribution table + default FAIL/POOR filter patch.

        Covers LLM template drift (ieval_external-20260915/20260916 user feedback):
        - The first Case-detail column, "case id", shows a truncated trace_id (first 3 + ... + last 3).
          Hover shows the full trace id; a click opens the detail modal on this page (no new tab).
          An output column is added after the query column
          (display_output, truncated to 40 chars, may wrap, hover shows the full text). The action column is dropped.
        - One "eval dimensions" card: the dimension column is narrowed, and the gap to the weight column is ≤ 3ch.
          The radar is centered and a "today/yesterday" legend is forced in (covers a missing legend and one deleted when cards merge).
        - Overall-attribution distribution → a display-name table (ieval=primary_issue_category;
          search_eval=deduction dimension; category|bar|count|percent).
        - Case details show only FAIL+POOR by default; other tiers appear only when chosen in the status dropdown.
        """
        if 'id="report-ux-patch"' in html:
            return html
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            return html[: body_close.start()] + _REPORT_UX_PATCH + html[body_close.start() :]
        return html + _REPORT_UX_PATCH

    @staticmethod
    def _inject_report_enhance(html: str) -> str:
        """Inject the enhancement-sections patch (Banner / waterfall / period delta / heatmap).

        Recognize and restyle an existing native section, and build a module only when it
        is missing, so the same data is not rendered twice. Data is read from
        REPORT.score_loss_waterfall / criterion_deltas / root_cause_dim_matrix.
        """
        if 'id="report-enhance-patch"' in html:
            return html
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            return html[: body_close.start()] + _REPORT_ENHANCE_PATCH + html[body_close.start() :]
        return html + _REPORT_ENHANCE_PATCH

    @staticmethod
    def _inject_dim_scores_layout(html: str) -> str:
        """Inject the per-dimension-scores vertical-layout patch before ``</body>``.

        LLM templates often wrap that column with a newline plus white-space:pre-line, but
        a td ID-selector nowrap rule has higher specificity and overrides it, so dimensions
        sit on one line. The patch finds the per-dimension-scores column by header text and
        uses inline style to force each dimension onto its own line (idempotent; data is unchanged).
        """
        if 'id="dim-scores-layout-patch"' in html:
            return html
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            return (
                html[: body_close.start()] + _DIM_SCORES_LAYOUT_PATCH + html[body_close.start() :]
            )
        return html + _DIM_SCORES_LAYOUT_PATCH

    @staticmethod
    def _inject_trace_link_patch(html: str) -> str:
        """Inject the "Trace link" caller-fill patch before ``</body>``.

        The webdev-agent external-link red line is a semantic judgment. Any prompt
        instruction to create a clickable external link (a static <a>, a DOM API, or
        window.open) rejects the whole response (search_eval-20260914-q3: a fixed
        rejection phrase instead of HTML). The prompt asks the LLM only for a plain-text
        slot column, and this patch injects the real <a> per row from cases-data after
        the page is generated (same split as _fill_link_slots). Idempotent; if the LLM
        did not generate the column, skip it and do not block.
        """
        if 'id="trace-link-patch"' in html:
            return html
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            return html[: body_close.start()] + _TRACE_LINK_PATCH + html[body_close.start() :]
        return html + _TRACE_LINK_PATCH

    @staticmethod
    def _inject_safe_insert_before_patch(html: str) -> str:
        """Inject the insertBefore reference-node fallback (before ``</body>``, idempotent).

        LLM chart templates often make the tooltip a sibling of the mount container, or
        call insertBefore before the createElement node is in the DOM. When the reference
        is not a child of the container the native call throws NotFoundError, the canvas
        never mounts, and the chart is blank (same root cause as the blank radars in
        ieval_external-20260915 / search_eval-20260915). The runtime wrap degrades to append.
        """
        if 'id="safe-insert-before-patch"' in html:
            return html
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            patch = _SAFE_INSERT_BEFORE_PATCH
            return html[: body_close.start()] + patch + html[body_close.start() :]
        return html + _SAFE_INSERT_BEFORE_PATCH

    @staticmethod
    def _fix_chart_tooltip_nesting(html: str) -> str:
        """Move a tooltip div that is a sibling of chart-box into the box (idempotent).

        The LLM may write ``#radar-tooltip`` as a sibling of ``#radar-box``, while JS
        mounts the canvas with ``box.insertBefore(canvas, tooltip)``. When the tooltip
        is not a child of the box this throws ``NotFoundError``, the canvas is never
        inserted, and the chart is blank (root cause of the blank radar in
        ieval_external-20260915; waterfall/trend tooltips sit inside the box, so they
        were not hit). Only the "empty box + empty tooltip" shape is handled; other
        structures are left alone.

        Tooltip id pairing is relaxed: prefer ``{name}-tooltip``, and on a miss drop
        the last word of ``name`` and try again (LLM naming variants such as
        ``radar-chart-box`` ↔ ``radar-tooltip`` — the same-root-cause incident in
        search_eval-20260915).
        """
        box_re = re.compile(
            r'(?P<open><div\b[^>]*\bid="(?P<name>[a-z0-9_-]+)-box"[^>]*>)'
            r"(?P<inner>\s*)</div>",
            re.IGNORECASE,
        )

        def _tip_ids(name: str) -> list[str]:
            ids = [f"{name}-tooltip"]
            parts = name.split("-")
            while len(parts) > 1:
                parts.pop()
                ids.append("-".join(parts) + "-tooltip")
            return ids

        def _find_tip(name: str, html_tail: str) -> re.Match[str] | None:
            for tid in _tip_ids(name):
                m = re.search(
                    rf'<div\b[^>]*\bid="{re.escape(tid)}"[^>]*>\s*</div>',
                    html_tail,
                    re.IGNORECASE,
                )
                if m is not None:
                    return m
            return None

        skipped: set[str] = set()
        while True:
            box_m = None
            for m in box_re.finditer(html):
                name = m.group("name")
                if name in skipped:
                    continue
                if _find_tip(name, html[m.end() :]) is not None:
                    box_m = m
                    break
            if box_m is None:
                return html
            name = box_m.group("name")
            tip_m = _find_tip(name, html[box_m.end() :])
            if tip_m is None:
                # tooltip is non-empty or the shape does not match — do not guess, skip this box
                skipped.add(name)
                continue
            tip_html = tip_m.group(0)
            tip_start = box_m.end() + tip_m.start()
            tip_end = box_m.end() + tip_m.end()
            insert_at = box_m.end("open")
            html = html[:insert_at] + tip_html + html[insert_at:tip_start] + html[tip_end:]

    @staticmethod
    def _isolate_init_steps(html: str) -> str:
        """Wrap top-level render calls inside ``function init(){...}`` in try/catch (idempotent).

        The prompt says one step failing must not stop later steps, but the LLM often
        calls them in sequence. Any throw aborts init and later sections stay blank
        (ieval_external-20260915: renderRadar threw NotFoundError, so renderAttribution
        never ran and the failed-attribution block was blank). Only a whole line that is
        one no-argument call with a render/draw/apply/load prefix is wrapped. A wrapped
        line becomes ``try{...}`` and no longer matches, so this stays idempotent.
        """
        m = re.search(r"function\s+init\s*\(\s*\)\s*\{", html)
        if not m:
            return html
        depth = 0
        i = m.end() - 1  # '{' that points at init
        body_start = i + 1
        while i < len(html):
            ch = html[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if i >= len(html):
            return html
        body = html[body_start:i]

        def _wrap(mm: re.Match[str]) -> str:
            indent, name = mm.group(1), mm.group(2)
            return (
                f"{indent}try{{{name}();}}catch(e){{"
                f"console.error('[agent-eval] init step failed: {name}', e);}}"
            )

        fixed = re.sub(
            r"(?m)^(\s*)((?:render|draw|apply|load)[A-Za-z0-9_$]*)\(\s*\);\s*$",
            _wrap,
            body,
        )
        return html[:body_start] + fixed + html[i:]

    @staticmethod
    def _fix_iife_global_scope(html: str) -> str:
        """Expose IIFE-scoped functions to ``window`` for inline event handlers.

        The LLM may wrap all JavaScript in an IIFE (Immediately Invoked
        Function Expression) — a common best practice. However, inline
        event handlers (``onclick="openDetail(...)"`` etc.) resolve
        function names in the **global scope**, where IIFE-scoped
        functions are invisible. This causes ``ReferenceError: X is not
        defined`` when users click Case ID links or detail buttons.

        This method detects the IIFE pattern and, if inline event handlers
        reference functions defined inside the IIFE, appends
        ``window.funcName = funcName;`` assignments before the IIFE's
        closing ``})();`` — making the functions accessible to inline
        handlers without otherwise changing the code.

        If no IIFE is found or no inline handlers reference IIFE-scoped
        functions, the HTML is returned unchanged.
        """
        # 1. Find all function names referenced in inline on* attributes.
        on_funcs: set[str] = set()
        for m in _ON_EVENT_FUNC_RE.finditer(html):
            func_name = m.group(1).strip()
            if func_name and func_name.isidentifier():
                on_funcs.add(func_name)

        if not on_funcs:
            return html  # No inline handlers → nothing to fix.

        # 2. Find JS <script> blocks (skip type="application/json" tags)
        #    that contain an IIFE opening.
        js_iife_scripts: list[tuple[int, int]] = []  # (content_start, content_end)
        for m in _SCRIPT_TAG_RE.finditer(html):
            attrs = m.group(1)
            content = m.group(2)
            if "application/json" in attrs.lower():
                continue  # Skip JSON data tags.
            if _IIFE_OPEN_RE.search(content):
                js_iife_scripts.append((m.start(2), m.end(2)))

        if not js_iife_scripts:
            return html  # No IIFE → functions are already global.

        # 3. Find all named function definitions in the HTML.
        defined_funcs = {m.group(1) for m in _FUNC_DEF_RE.finditer(html)}

        # 4. Only expose functions that are both defined AND referenced.
        to_expose = on_funcs & defined_funcs
        if not to_expose:
            return html

        # 5. Build the window assignment string.
        assignments = "; ".join(f"window.{f}={f}" for f in sorted(to_expose))

        # 6. For each IIFE script, insert assignments before the IIFE
        #    closing })(); . Process from last to first to preserve offsets.
        result = html
        for content_start, content_end in reversed(js_iife_scripts):
            content = result[content_start:content_end]
            close_matches = list(_IIFE_CLOSE_RE.finditer(content))
            if not close_matches:
                continue
            last_close = close_matches[-1]
            insert_offset = content_start + last_close.start()
            result = result[:insert_offset] + assignments + "; " + result[insert_offset:]

        logger.info(
            "webdev_client.fix_iife_scope exposed_functions=%s",
            sorted(to_expose),
        )

        return result

    @staticmethod
    def _fix_render_table_row_bind(html: str) -> str:
        """Change ``rows[idx].addEventListener(...) })(i)`` to ``})(ri)``.

        ieval_external-20260916: after ``tbody.innerHTML=h`` the template used the ``i``
        left by the build loop (equal to ``page.length``) to read ``rows[idx]``.
        ``undefined.addEventListener`` threw TypeError, the renderTable wrap chain
        stopped, and case-id hover/click plus the query/output enhancements were all skipped.
        """
        return _RENDER_TABLE_ROW_BIND_RE.sub(r"\1ri\2", html, count=1)

    @staticmethod
    def _fix_undefined_function_calls(html: str) -> str:
        """Inject an alias or stub for a function that is called but not defined, so init does not stop the whole page.

        An LLM template may rename a function and leave the call site stale (init() calls
        ``renderTrendChart()`` but the definition is ``drawTrendChart()``). The
        ``ReferenceError`` aborts init, and the dimension table / attribution / Case
        details after it never render (online regression in ieval_external-20260908-q1).

        For each bare call that is not defined (excluding keywords, builtins, method
        calls, and calls guarded by ``typeof X``):
        1. If the defined functions have a unique same-suffix candidate (verb prefixes
           such as render/draw/show swapped, for example renderTrendChart → drawTrendChart),
           inject an alias wrapped in try/catch. If the target is unreachable (an IIFE
           local, for example) it only console.errors and does not abort init.
        2. Otherwise inject a console.error stub so init continues with later render steps.

        The declaration is inserted at the top of the ``<script>`` block that contains
        the call (function declarations are hoisted and usable anywhere in the block).
        If later code or a later script defines a real function of the same name, it
        overrides this declaration and behavior is unchanged. A name already injected
        is "defined" on the next scan, so this is naturally idempotent.
        """
        script_matches = list(_SCRIPT_TAG_RE.finditer(html))
        if not script_matches:
            return html

        defined: set[str] = set()
        injections: list[tuple[int, str]] = []  # (script_content_start, code)
        total = 0

        for sm in script_matches:
            attrs = sm.group(1)
            if "application/json" in attrs.lower():
                continue
            content = sm.group(2)
            # Strip string literals and comments before scanning, so their text is not mistaken for a function call.
            scan = _JS_COMMENT_RE.sub("", _JS_STRING_RE.sub("", content))

            # Definitions in this block (function declarations are hoisted and visible to the whole block).
            block_defined = set(_FUNC_DEF_RE.findall(scan))
            block_defined.update(m.group(1) for m in _WINDOW_FUNC_ASSIGN_RE.finditer(scan))
            block_defined.update(_VAR_FUNC_DEF_RE.findall(scan))

            pool = defined | block_defined
            undefined = sorted(
                name
                for name in (m.group(1) for m in _BARE_CALL_RE.finditer(scan))
                if name not in pool and name not in _JS_CALL_IGNORE and f"typeof {name}" not in scan
            )

            lines: list[str] = []
            for name in undefined:
                alias = _unique_suffix_alias(name, pool)
                if alias:
                    lines.append(
                        f"function {name}(){{try{{return {alias}.apply(this,arguments);}}"
                        f"catch(e){{console.error("
                        f"'[agent-eval] alias {name}->{alias} failed',e);}}}}"
                    )
                else:
                    lines.append(
                        f"function {name}(){{console.error("
                        f"'[agent-eval] {name} is not defined — stubbed');}}"
                    )
            if lines:
                injections.append(
                    (
                        sm.start(2),
                        "/* agent-eval: undefined-call guard */\n" + "\n".join(lines) + "\n",
                    )
                )
                total += len(lines)

            defined |= block_defined

        if not injections:
            return html

        result = html
        for start, code in reversed(injections):
            result = result[:start] + code + result[start:]
        logger.info("webdev_client.fix_undefined_calls injected=%d", total)
        return result

    @staticmethod
    def _find_matching_brace(text: str, open_pos: int) -> int | None:
        """Find the position of ``}`` matching the ``{`` at *open_pos*.

        Uses a simple brace counter that is aware of single/double/backtick
        strings and escape sequences.  Comments (``//`` and ``/* */``) are
        not handled — in practice LLM-generated report JS is clean enough.
        """
        depth = 0
        i = open_pos
        in_str: str | None = None
        while i < len(text):
            ch = text[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in ('"', "'", "`"):
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    @staticmethod
    def _fix_window_function_hoisting(html: str) -> str:
        """Convert ``window.X = function() {}`` to hoisted function declarations.

        When the LLM defines functions as ``window.X = function() { ... }``
        inside an IIFE, the functions are only available *after* the
        assignment executes.  If the code calls ``X()`` before the
        assignment (which is common when the LLM puts setup/initialisation
        code before function definitions), it crashes with
        ``TypeError: X is not a function``, leaving the case detail table
        body empty.

        This method converts ``window.X = function(params) { body }`` to
        ``function X(params) { body }\nwindow.X = X;`` — the function
        declaration is hoisted to the top of the enclosing scope, making
        it available throughout.  The ``window.X = X`` assignment keeps it
        globally accessible for inline event handlers.

        If no ``window.X = function`` patterns are found, the HTML is
        returned unchanged.
        """
        total_converted = 0
        result = html

        script_matches = list(_SCRIPT_TAG_RE.finditer(result))
        for sm in reversed(script_matches):
            attrs = sm.group(1)
            if "application/json" in attrs.lower():
                continue
            content_start = sm.start(2)
            content_end = sm.end(2)
            content = result[content_start:content_end]

            # Collect (start, end, replacement) tuples for each
            # window.X = function(params) { body } pattern.
            replacements: list[tuple[int, int, str]] = []
            for fm in _WINDOW_FUNC_ASSIGN_RE.finditer(content):
                func_name = fm.group(1)
                params = fm.group(2).strip()
                # The opening ``{`` is the last char of the regex match.
                brace_pos = fm.end() - 1
                if content[brace_pos] != "{":
                    continue
                close_pos = WebdevReportClient._find_matching_brace(
                    content,
                    brace_pos,
                )
                if close_pos is None:
                    continue
                body = content[brace_pos + 1 : close_pos]
                full_start = fm.start()
                full_end = close_pos + 1
                replacement = (
                    f"function {func_name}({params}) {{{body}}}\nwindow.{func_name}={func_name};"
                )
                replacements.append((full_start, full_end, replacement))

            if not replacements:
                continue

            # Apply from last to first to preserve offsets.
            new_content = content
            for start, end, repl in reversed(replacements):
                new_content = new_content[:start] + repl + new_content[end:]

            result = result[:content_start] + new_content + result[content_end:]
            total_converted += len(replacements)

        if total_converted:
            logger.info(
                "webdev_client.fix_window_hoisting converted=%d",
                total_converted,
            )

        return result

    def _log_send_spec_summary(
        self,
        request_id: str,
        summary: "_SendSpecSummary",
    ) -> None:
        """Log before send: spec / prompt / payload sizes and a sample of the first case."""
        spec = summary.spec
        prompt = summary.prompt
        payload = summary.payload
        case_count = len(spec.get("case_results", []))
        criteria_count = len(spec.get("criteria", []))
        crit_metrics = spec.get("criterion_metrics", {})
        attr_metrics = spec.get("attribution_metrics", {})
        spec_size = len(json.dumps(spec, ensure_ascii=False))
        prompt_size = len(prompt)
        payload_size = len(json.dumps(payload, ensure_ascii=False))
        logger.info(
            "webdev_client.send_spec request_id=%s url=%s "
            "benchmark_name=%s service_name=%s run_id=%s "
            "spec_keys=%s spec_size=%d prompt_size=%d payload_size=%d "
            "case_results=%d criteria=%d "
            "criterion_metrics_keys=%s attribution_metrics_keys=%s "
            "final_score=%s total_samples=%s",
            request_id,
            self._url,
            spec.get("benchmark_name"),
            spec.get("service_name"),
            spec.get("run_id"),
            list(spec.keys()),
            spec_size,
            prompt_size,
            payload_size,
            case_count,
            criteria_count,
            list(crit_metrics.keys()),
            list(attr_metrics.keys()),
            spec.get("final_score"),
            spec.get("total_samples"),
        )
        if case_count == 0:
            logger.warning(
                "webdev_client.no_case_results request_id=%s case_results missing or empty in spec",
                request_id,
            )
            return
        first_case = spec["case_results"][0]
        logger.info(
            "webdev_client.case_sample request_id=%s "
            "first_case_id=%s first_case_keys=%s "
            "first_case_scores_keys=%s",
            request_id,
            first_case.get("case_id"),
            list(first_case.keys()),
            list(first_case.get("scores", {}).keys())
            if isinstance(first_case.get("scores"), dict)
            else None,
        )

    @staticmethod
    def _log_response_structure(request_id: str, data: dict[str, Any]) -> None:
        """Log after the response: jsonrpc artifact structure and the error channel."""
        result_obj = data.get("result", {})
        artifacts = result_obj.get("artifacts", [])
        artifact_count = len(artifacts)
        parts_info: list[str] = []
        for i, art in enumerate(artifacts):
            parts = art.get("parts", [])
            for j, p in enumerate(parts):
                kind = p.get("kind", "?")
                text_len = len(p.get("text", "")) if kind == "text" else 0
                parts_info.append(f"artifact[{i}].parts[{j}] kind={kind} text_len={text_len}")
        logger.info(
            "webdev_client.response_struct request_id=%s "
            "jsonrpc=%s method=%s artifact_count=%d parts=[%s]",
            request_id,
            data.get("jsonrpc"),
            data.get("method"),
            artifact_count,
            ", ".join(parts_info) if parts_info else "(none)",
        )
        if "error" in data:
            logger.error(
                "webdev_client.jsonrpc_error request_id=%s error=%s",
                request_id,
                json.dumps(data["error"], ensure_ascii=False)[:500],
            )

    @staticmethod
    def enrich_report_spec(spec: dict[str, Any]) -> None:
        """Shared precomputation entry for a report (the static-template path and the LLM path must share it).

        Order and dependencies (do not reorder):
        1. ``_attach_grade_distribution`` — tier counts
        2. ``_attach_today_criterion_means`` — flat means for the radar "today" series
        3. ``_attach_root_cause_display`` — Case root-cause display column
        4. ``_attach_report_insights`` — depends on root_cause_display
        5. ``_attach_enhance_modules`` — depends on root_cause + today means
           (waterfall / heatmap / dimension period delta)

        A new ``_attach_*`` step is added only in this method. Callers must not inline a copy of the sequence.
        """
        WebdevReportClient._attach_grade_distribution(spec)
        WebdevReportClient._attach_today_criterion_means(spec)
        WebdevReportClient._attach_root_cause_display(spec)
        WebdevReportClient._attach_report_insights(spec)
        WebdevReportClient._attach_enhance_modules(spec)

    @staticmethod
    def _attach_grade_distribution(spec: dict[str, Any]) -> None:
        """Count the rubric_v2 grade distribution from case_results and inject it into the spec (idempotent).

        Each case's ``grade`` (EXCELLENT/GOOD/ACCEPTABLE/POOR/FAIL) is precomputed on the
        agent-eval side by ``load_case_results`` from ``scores.fused.score``
        (``rubric_v2_grade``). This only counts — ``{grade: count}``. A case with no
        grade (missing fused row / evaluation failure) is not counted. A non-empty
        ``grade_distribution`` already set on the spec is not overwritten. The report's
        overall-results column renders from this precomputed source, and JS does not
        derive the thresholds again.
        """
        if spec.get("grade_distribution"):
            return
        dist: dict[str, int] = {}
        for case in spec.get("case_results") or []:
            if not isinstance(case, dict):
                continue
            grade = case.get("grade")
            if isinstance(grade, str) and grade:
                dist[grade] = dist.get(grade, 0) + 1
        if dist:
            spec["grade_distribution"] = dist

    @staticmethod
    def _attach_today_criterion_means(spec: dict[str, Any]) -> None:
        """Extract today's per-dimension means from criterion_metrics into a flat map (idempotent).

        For the radar "today" dataset only: ``{criterion_id: mean}``, the same shape as
        ``yesterday_criterion_means``, so the frontend can read both with one getter.
        This stops the LLM from dividing the nested criterion_metrics stats object as if
        it were a number (object/number = NaN, the canvas draws nothing, and there is no
        console error — the root cause of the blank online radar in search_eval-20260914-q4).
        """
        if spec.get("today_criterion_means"):
            return
        means: dict[str, float] = {}
        for cid, stats in (spec.get("criterion_metrics") or {}).items():
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean")
            # bool is a subclass of int; exclude it explicitly (True/False are not scores).
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                means[str(cid)] = float(mean)
        if means:
            spec["today_criterion_means"] = means

    # PLR0912 waiver: nested condition skips while walking cases are structural branches of data extraction.
    @staticmethod
    def _attach_report_insights(spec: dict[str, Any]) -> None:  # noqa: PLR0912
        """Compute report insight data from case_results and inject ``insight_text``.

        **Every report generation recomputes from the current results** (overwriting an
        existing ``insight_text``), so the insight banner stays consistent with the
        completion rate / overall score / root-cause loss total. A high completion rate
        with a low overall score, the Top2 root-cause share, and the improvement
        suggestion all follow this run's cases.

        The root-cause scale matches the Case root-cause column and the waterfall:
        prefer ``root_cause_display``; if missing, fall back to the attribution category
        (through value_labels plus the display-name map). Contribution is ranked by the
        **fused loss total** (not the per-case mean).
        """
        case_results = spec.get("case_results")
        if not case_results:
            return

        labels = WebdevReportClient._root_cause_value_labels(spec)
        cat_labels = labels.get("primary_issue_category") or {}
        full_mark = _score_full_mark(spec)
        # {display_label: [fused_loss, ...]}
        rc_losses: dict[str, list[float]] = {}
        for case in case_results:
            if not isinstance(case, dict):
                continue
            display = case.get("root_cause_display")
            if not isinstance(display, str) or not display.strip():
                attr = case.get("attribution")
                if not isinstance(attr, dict):
                    scores = case.get("scores")
                    fused = scores.get("fused") if isinstance(scores, dict) else None
                    attr = fused.get("attribution") if isinstance(fused, dict) else None
                if not isinstance(attr, dict):
                    continue
                raw = attr.get("primary_issue_category")
                if not raw or not isinstance(raw, str):
                    continue
                mapped = cat_labels.get(raw, raw)
                display = _root_cause_cn(mapped) or mapped
            display = display.strip()
            if not display or display.lower() in {"none", "null"}:
                continue
            scores = case.get("scores")
            if not isinstance(scores, dict):
                continue
            fused = scores.get("fused")
            fused_score = fused.get("score") if isinstance(fused, dict) else None
            if not isinstance(fused_score, (int, float)) or isinstance(fused_score, bool):
                continue
            fused_loss = min(full_mark, max(0.0, full_mark - float(fused_score)))
            rc_losses.setdefault(display, []).append(fused_loss)

        if not rc_losses:
            # When there is no usable root-cause loss, clear the old copy so a leftover does not mislead.
            spec.pop("insight_text", None)
            return

        # Total score loss (positive, descending) — the same scale as the waterfall weights.
        score_loss = {rc: round(sum(v), 3) for rc, v in rc_losses.items()}
        score_loss = dict(sorted(score_loss.items(), key=lambda x: x[1], reverse=True))

        insight = WebdevReportClient._generate_insight_text(spec, score_loss)
        if insight:
            spec["insight_text"] = insight
        else:
            spec.pop("insight_text", None)

    @staticmethod
    def _attach_root_cause_display(spec: dict[str, Any]) -> None:
        """Precompute the root-cause column text ``root_cause_display`` for each case (idempotent).

        Branches by business (the scale clarified with the user):

        - **ieval** (benchmark_name starts with ieval): use ``primary_issue_layer`` /
          ``primary_issue_category`` from the attribution data (``value_labels`` to a
          display label, dedupe the prefix, then the display-name map, aligned with the
          overall-attribution table). With no attribution, fall back to the display names
          of ``scores[cid]`` dimensions that are not full score.
        - **search_eval**: use the display names (criteria names) of every ``sub_scores``
          dimension in the judge that is not full score. Do not read attribution.
        - **Other businesses**: use the attribution primary ``root_cause`` text. With no
          attribution (or no primary cause), fall back to display names of dimensions that
          are not full score (prefer ``scores[cid]``, then ``sub_scores``).

        The full-score scale matches the report: a dimension whose id starts with d0/D0,
        or whose name contains the safety keyword, has full score 1; the rest are 2.
        Missing or non-numeric dimensions are not counted. All full scores, or no usable
        data, means the field is not set (the frontend column shows a muted "-").

        Attribution location: prefer case-level ``attribution``, and fall back to
        ``scores.fused.attribution``. A case that already has ``root_cause_display`` is
        not overwritten (idempotent).
        """
        criteria = spec.get("criteria") or []
        names = {
            str(c["id"]): str(c.get("name") or c["id"])
            for c in criteria
            if isinstance(c, dict) and c.get("id")
        }
        labels = WebdevReportClient._root_cause_value_labels(spec)
        bm = WebdevReportClient._root_cause_benchmark_type(spec)
        for case in spec.get("case_results") or []:
            if not isinstance(case, dict) or case.get("root_cause_display"):
                continue
            display = WebdevReportClient._resolve_case_root_cause_display(
                case,
                names,
                labels,
                benchmark_type=bm,
            )
            if display:
                case["root_cause_display"] = display

    @staticmethod
    def _root_cause_benchmark_type(spec: dict[str, Any]) -> str:
        """Business branch of the root-cause column: ``ieval`` / ``search_eval`` / ``other``."""
        name = str(spec.get("benchmark_name") or "").lower()
        if name.startswith("ieval"):
            return "ieval"
        if name == "search_eval" or name.startswith("search_eval"):
            return "search_eval"
        return "other"

    @staticmethod
    def _attach_enhance_modules(spec: dict[str, Any]) -> None:
        """Idempotently inject enhancement-section data (add fields only; do not change existing fields).

        Added:
        - ``criterion_deltas``: per-dimension period-delta list
          ``[{id, name, today, yesterday, delta}, ...]``
        - ``score_loss_waterfall``: score-loss waterfall
          ``{start, end, steps: [{label, loss, count}, ...]}``
        - ``root_cause_dim_matrix``: root-cause × dimension loss matrix
          ``{rows, cols: [{id,name}], cells: {row: {cid: loss}}}``

        Dependency: call this after ``_attach_root_cause_display`` /
        ``_attach_today_criterion_means`` (it uses the display root cause and today's means).
        """
        WebdevReportClient._attach_criterion_deltas(spec)
        WebdevReportClient._attach_score_loss_waterfall(spec)
        WebdevReportClient._attach_root_cause_dim_matrix(spec)

    @staticmethod
    def _attach_criterion_deltas(spec: dict[str, Any]) -> None:
        """Inject ``criterion_deltas`` (today vs yesterday); skip when yesterday's data is missing."""
        if spec.get("criterion_deltas"):
            return
        today = spec.get("today_criterion_means") or {}
        yest = spec.get("yesterday_criterion_means") or {}
        if not isinstance(today, dict) or not isinstance(yest, dict) or not yest:
            return
        criteria = spec.get("criteria") or []
        deltas: list[dict[str, Any]] = []
        for c in criteria:
            if not isinstance(c, dict) or not c.get("id"):
                continue
            cid = str(c["id"])
            t = today.get(cid)
            y = yest.get(cid)
            if not isinstance(t, (int, float)) or not isinstance(y, (int, float)):
                continue
            if isinstance(t, bool) or isinstance(y, bool):
                continue
            deltas.append(
                {
                    "id": cid,
                    "name": str(c.get("name") or cid),
                    "today": round(float(t), 4),
                    "yesterday": round(float(y), 4),
                    "delta": round(float(t) - float(y), 4),
                },
            )
        if deltas:
            spec["criterion_deltas"] = deltas

    @staticmethod
    def _attach_score_loss_waterfall(spec: dict[str, Any]) -> None:
        """Inject ``score_loss_waterfall`` (the overall-score gap allocated by root-cause weight).

        Scale: total gap ``Δ = score_full_mark − final_score`` (default full score 2.0;
        the trace daily report passes ``score_full_mark`` through). Each root cause's
        weight is its cases' fused loss total divided by the fused loss total of all
        valid cases. Cases with no root cause count as unattributed, so their loss is
        not wrongly allocated to a known root cause. Bar height is ``loss = Δ × weight``.
        The bars sum to about Δ, and each bar is at most the full score.
        """
        if spec.get("score_loss_waterfall"):
            return
        case_results = spec.get("case_results") or []
        if not case_results:
            return
        full_mark = _score_full_mark(spec)
        # {label: [per-case fused loss, ...]}
        buckets: dict[str, list[float]] = {}
        for case in case_results:
            if not isinstance(case, dict):
                continue
            raw_label = case.get("root_cause_display")
            label = (
                raw_label.strip()
                if isinstance(raw_label, str) and raw_label.strip()
                else _UNATTRIBUTED_ROOT_CAUSE_LABEL
            )
            scores = case.get("scores")
            fused = scores.get("fused") if isinstance(scores, dict) else None
            fused_score = fused.get("score") if isinstance(fused, dict) else None
            if not isinstance(fused_score, (int, float)) or isinstance(fused_score, bool):
                continue
            fused_loss = min(full_mark, max(0.0, full_mark - float(fused_score)))
            buckets.setdefault(label, []).append(fused_loss)
        if not buckets:
            return
        raw_totals = {label: sum(vals) for label, vals in buckets.items()}
        total_raw = sum(raw_totals.values())
        if total_raw <= 0:
            return
        end = spec.get("final_score")
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            end = 0.0
        end_f = float(end)
        delta = min(full_mark, max(0.0, full_mark - end_f))
        steps = [
            {
                "label": label,
                "loss": round(delta * (raw_totals[label] / total_raw), 3),
                "count": len(buckets[label]),
                # Keep the raw accumulated loss for investigation; it does not affect bar height.
                "raw_loss": round(raw_totals[label], 3),
            }
            for label in raw_totals
        ]
        steps.sort(key=lambda x: x["loss"], reverse=True)
        # The waterfall shows at most 7 groups. "Unattributed" is always its own column, so "other" does not hide it.
        unattributed = next(
            (s for s in steps if s["label"] == _UNATTRIBUTED_ROOT_CAUSE_LABEL),
            None,
        )
        attributed = [s for s in steps if s["label"] != _UNATTRIBUTED_ROOT_CAUSE_LABEL]
        attributed_limit = 5 if unattributed else 6
        top = attributed[:attributed_limit]
        if unattributed:
            top.append(unattributed)
        rest = attributed[attributed_limit:]
        if rest:
            top.append(
                {
                    "label": "其他",
                    "loss": round(sum(s["loss"] for s in rest), 3),
                    "count": sum(s["count"] for s in rest),
                    "raw_loss": round(sum(s["raw_loss"] for s in rest), 3),
                },
            )
        top.sort(key=lambda x: x["loss"], reverse=True)
        spec["score_loss_waterfall"] = {
            "start": full_mark,
            "end": round(end_f, 4),
            "delta": round(delta, 4),
            "steps": top,
        }

    @staticmethod
    def _attach_root_cause_dim_matrix(spec: dict[str, Any]) -> None:
        """Inject ``root_cause_dim_matrix`` (root-cause × dimension **mean** loss heatmap).

        A valid case with no ``root_cause_display`` goes into unattributed. A cell is
        ``sum(full score − score) / the count of that root cause's cases that have a score
        on that dimension`` (full-score cases are included, loss=0 stays in the denominator),
        so it reflects that root cause's **typical** loss on that dimension. Each cell is
        at most that dimension's full score (1 for the safety dimension, 2 for the rest).
        """
        if spec.get("root_cause_dim_matrix"):
            return
        criteria = spec.get("criteria") or []
        cols = [
            {"id": str(c["id"]), "name": str(c.get("name") or c["id"])}
            for c in criteria
            if isinstance(c, dict) and c.get("id")
        ]
        if not cols:
            return
        names = {c["id"]: c["name"] for c in cols}
        # cells_sum[rc][cid] / cells_n[rc][cid] → mean
        cells_sum: dict[str, dict[str, float]] = {}
        cells_n: dict[str, dict[str, int]] = {}
        row_totals: dict[str, float] = {}
        # A passed score_full_mark overrides the default per-dimension full-score heuristic (including "1.0").
        # Default stays 1 for safety and 2 otherwise, so an unconfigured dimension is not the overall full score.
        mark_override = spec.get("score_full_mark")
        use_mark = _score_full_mark(spec) if mark_override is not None else None
        for case in spec.get("case_results") or []:
            if not isinstance(case, dict):
                continue
            raw_label = case.get("root_cause_display")
            label = (
                raw_label.strip()
                if isinstance(raw_label, str) and raw_label.strip()
                else _UNATTRIBUTED_ROOT_CAUSE_LABEL
            )
            scores = case.get("scores")
            if not isinstance(scores, dict):
                continue
            row_s = cells_sum.setdefault(label, {})
            row_n = cells_n.setdefault(label, {})
            for cid, cname in names.items():
                cdata = scores.get(cid)
                if not isinstance(cdata, dict):
                    continue
                score = _numeric_score(cdata.get("score"))
                if score is None:
                    continue
                # Full scores (loss=0) stay in the denominator. Otherwise the mean is only over cases that already lost
                # points, and 1 of 3 scoring 0 would show 2.0, exaggerating that root cause's typical loss.
                full_score = use_mark if use_mark is not None else _dimension_full_score(cid, cname)
                loss = min(full_score, max(0.0, full_score - score))
                row_s[cid] = row_s.get(cid, 0.0) + loss
                row_n[cid] = row_n.get(cid, 0) + 1
                row_totals[label] = row_totals.get(label, 0.0) + loss
        if not cells_sum:
            return
        ranked_rows = sorted(row_totals.keys(), key=lambda k: row_totals[k], reverse=True)
        rows = ranked_rows[:8]
        if (
            _UNATTRIBUTED_ROOT_CAUSE_LABEL in ranked_rows
            and _UNATTRIBUTED_ROOT_CAUSE_LABEL not in rows
        ):
            rows = rows[:7] + [_UNATTRIBUTED_ROOT_CAUSE_LABEL]
        trimmed: dict[str, dict[str, float]] = {}
        for rc in rows:
            trimmed[rc] = {
                cid: round(cells_sum[rc][cid] / cells_n[rc][cid], 3)
                for cid in cells_sum[rc]
                if cells_n[rc].get(cid)
            }
        spec["root_cause_dim_matrix"] = {
            "rows": rows,
            "cols": cols,
            "cells": trimmed,
            "unit": "mean_loss",
        }

    @staticmethod
    def _validate_case_required_fields(spec: dict[str, Any]) -> list[str]:
        """Check that required Case-detail columns have values (warning only; do not block the upload).

        The five columns query / case status / overall score / per-dimension scores / Trace link
        must have a value on every case.
        """
        criteria = spec.get("criteria") or []
        criterion_ids = [str(c["id"]) for c in criteria if isinstance(c, dict) and c.get("id")]
        problems: list[str] = []
        for case in spec.get("case_results") or []:
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id") or "?")
            missing: list[str] = []
            if not str(case.get("display_input") or "").strip():
                missing.append("query")
            if not case.get("grade"):
                missing.append("case状态")
            scores = case.get("scores")
            fused = scores.get("fused") if isinstance(scores, dict) else None
            if not isinstance(fused, dict) or _numeric_score(fused.get("score")) is None:
                missing.append("综合分")
            if criterion_ids:
                dim_missing = [
                    cid
                    for cid in criterion_ids
                    if not isinstance(scores, dict)
                    or not isinstance(scores.get(cid), dict)
                    or _numeric_score(scores[cid].get("score")) is None
                ]
                if dim_missing:
                    missing.append(f"各维度得分({','.join(dim_missing)})")
            if not str(case.get("trace_url") or "").strip():
                missing.append("Trace链接")
            if missing:
                problems.append(f"case {case_id} 缺失必有值列: {'/'.join(missing)}")
        return problems

    @staticmethod
    def _root_cause_value_labels(spec: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Collect ``attribution_metrics[*].value_labels``: {field name: {raw value: display label}}."""
        labels: dict[str, dict[str, str]] = {}
        for stats in (spec.get("attribution_metrics") or {}).values():
            if not isinstance(stats, dict):
                continue
            vl = stats.get("value_labels")
            if not isinstance(vl, dict):
                continue
            for field, mapping in vl.items():
                if isinstance(mapping, dict):
                    labels.setdefault(str(field), {}).update(
                        {str(k): str(v) for k, v in mapping.items()},
                    )
        return labels

    @staticmethod
    def _resolve_case_root_cause_display(
        case: dict[str, Any],
        names: dict[str, str],
        labels: dict[str, dict[str, str]],
        benchmark_type: str = "other",
    ) -> str | None:
        """Resolve one case's root-cause display text (branches: ``_attach_root_cause_display``).

        - ``benchmark_type="ieval"``: attribution layer/category (value_labels plus prefix
          dedupe) → display name; no attribution → display names of ``scores[cid]``
          dimensions that are not full score.
        - ``benchmark_type="search_eval"``: display names of ``sub_scores`` dimensions
          that are not full score (attribution is not read).
        - ``benchmark_type="other"``: the attribution primary ``root_cause`` text; no
          primary cause → display names of dimensions that are not full score (prefer
          ``scores[cid]``, then ``sub_scores``).
        """
        attr = case.get("attribution")
        if not isinstance(attr, dict):
            scores = case.get("scores")
            fused = scores.get("fused") if isinstance(scores, dict) else None
            attr = fused.get("attribution") if isinstance(fused, dict) else None

        if benchmark_type == "search_eval":
            dims = WebdevReportClient._underfull_dim_names(case, names, prefer="sub_scores")
            return "/".join(dims) if dims else None

        if benchmark_type == "ieval":
            if isinstance(attr, dict):
                parts = []
                for field in ("primary_issue_layer", "primary_issue_category"):
                    value = attr.get(field)
                    if value:
                        text = str(value)
                        parts.append(labels.get(field, {}).get(text, text))
                joined = WebdevReportClient._dedupe_root_cause_parts(parts)
                # none/empty means no valid attribution, so fall back to dimensions that are not full score.
                if joined and joined.lower() not in {"none", "null", ""}:
                    return _root_cause_cn(joined)
            dims = WebdevReportClient._underfull_dim_names(case, names, prefer="scores")
            return "/".join(dims) if dims else None

        # other: the attribution primary-cause text; if there is no primary cause, use dimensions that are not full score.
        if isinstance(attr, dict):
            root = attr.get("root_cause")
            if isinstance(root, str) and root.strip():
                return root.strip()
        dims = WebdevReportClient._underfull_dim_names(case, names, prefer="auto")
        return "/".join(dims) if dims else None

    @staticmethod
    def _dedupe_root_cause_parts(parts: list[str]) -> str:
        """Merge issue-layer and category display text, dropping a repeated prefix.

        value_labels often map the raw ``primary_issue_category`` to
        ``"{layer}/{category}"`` (for example strategy_error → planning/strategy_error).
        Joining that with ``primary_issue_layer`` yields ``planning/planning/strategy_error``
        (repeated prefix on the root-cause column of the ieval_external-20260915 online
        report). If a later segment starts with the previous segment plus "/", this method
        drops the previous segment and keeps the more specific one.
        """
        cleaned = [p for p in parts if p and str(p).strip()]
        if not cleaned:
            return ""
        out: list[str] = []
        for p in cleaned:
            s = str(p).strip()
            if out and s.startswith(out[-1] + "/"):
                out[-1] = s
            else:
                out.append(s)
        return "/".join(out)

    @staticmethod
    def _underfull_dim_names(
        case: dict[str, Any],
        names: dict[str, str],
        prefer: str = "auto",
    ) -> list[str]:
        """Display names of dimensions that are not full score (criteria order; keys criteria do not cover are appended).

        Data-source choice (the ``prefer`` argument):

        - ``"scores"``: read only ``case.scores[cid]`` (criteria dimensions, the same
          scale as the per-dimension scores column). ieval uses this scale.
        - ``"sub_scores"``: read only ``case.sub_scores`` (a flat {dimension key: score}
          shape). search_eval uses this scale.
        - ``"auto"`` (default): prefer ``scores[cid]``, and fall back to ``sub_scores``
          only when scores has no non-fused dimension. Other businesses use this scale.

        Background (ieval_external-20260915): ieval ``sub_scores`` are sub-dimensions of
        D6 (D6a/D6b) and are not in criteria (D0-D6). The old rule "non-empty sub_scores
        wins" made the root-cause fallback show the raw keys ``D6a/D6b`` instead of the
        display names of the dimensions that are actually not full score in the
        per-dimension scores column.
        """
        dim_scores: dict[str, float] = {}

        def read_scores() -> None:
            scores = case.get("scores")
            if isinstance(scores, dict):
                for cid, cdata in scores.items():
                    if str(cid).lower() == "fused" or not isinstance(cdata, dict):
                        continue
                    score = _numeric_score(cdata.get("score"))
                    if score is not None:
                        dim_scores[str(cid)] = score

        def read_sub_scores() -> None:
            sub = case.get("sub_scores")
            if isinstance(sub, dict) and sub:
                for key, val in sub.items():
                    score = _numeric_score(val.get("score") if isinstance(val, dict) else val)
                    if score is not None:
                        dim_scores[str(key)] = score

        if prefer == "sub_scores":
            read_sub_scores()
        elif prefer == "scores":
            read_scores()
        else:  # auto
            read_scores()
            if not dim_scores:
                read_sub_scores()

        ordered = [cid for cid in names if cid in dim_scores]
        ordered += [cid for cid in dim_scores if cid not in names]
        return [
            names.get(cid, cid)
            for cid in ordered
            if dim_scores[cid] < _dimension_full_score(cid, names.get(cid, ""))
        ]

    @staticmethod
    def _weakest_dimension_label(spec: dict[str, Any]) -> str | None:
        """Label of the dimension with the lowest full-score rate, for example a display name plus 1.31/2."""
        criteria = spec.get("criteria") or []
        metrics = spec.get("criterion_metrics") or {}
        if not isinstance(metrics, dict):
            return None
        ranked: list[tuple[float, str, float, float]] = []
        for raw in criteria:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            cid = str(raw["id"])
            name = str(raw.get("name") or cid)
            stats = metrics.get(cid)
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean")
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                continue
            full = _dimension_full_score(cid, name)
            if full <= 0:
                continue
            ranked.append((float(mean) / full, name, float(mean), full))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0])
        _ratio, name, mean, full = ranked[0]
        return f"「{name}」{mean:.2f}/{full:.0f}"

    @staticmethod
    def _clip_insight(text: str, max_chars: int = _INSIGHT_MAX_CHARS) -> str:
        """Truncate on a period or semicolon so the insight banner stays within ``max_chars`` characters."""
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        for sep in ("。", "；"):
            idx = cut.rfind(sep)
            if idx >= max_chars // 2:
                return text[: idx + 1]
        return cut.rstrip("；。，、") + "…"

    @staticmethod
    def _generate_insight_text(
        spec: dict[str, Any],
        score_loss: dict[str, float],
    ) -> str | None:
        """Build automatic insight text from metric rules (display language, at most 200 characters).

        ``score_loss`` is ``{root-cause display name: loss total (positive)}``, already
        sorted by total descending. Rule template:
        1. High completion rate (>90%) but a low overall score → a quality gap (planning/answering), not a broken flow
        2. Each Top2 root cause's share plus their combined share
        3. The weakest dimension (lowest full-score rate)
        4. An improvement suggestion by root-cause type (accepts both display-name and English root-cause names)
        """
        if not score_loss:
            return None
        final_score = spec.get("final_score")
        total_samples = spec.get("total_samples", 0)
        total_success = spec.get("total_success", 0)
        if not isinstance(final_score, int | float) or not total_samples:
            return None

        completion_rate = total_success / total_samples * 100 if total_samples else 0
        full_mark = _score_full_mark(spec)
        score_pct = float(final_score) / full_mark * 100 if full_mark > 0 else 0.0
        total_loss = sum(abs(v) for v in score_loss.values())

        parts: list[str] = []

        # Completion-rate vs score insight.
        if completion_rate >= _INSIGHT_HIGH_COMPLETION_RATE and score_pct < _INSIGHT_LOW_SCORE_PCT:
            gap = completion_rate - score_pct
            parts.append(
                f"任务完成率 {completion_rate:.1f}% 但综合分仅 {score_pct:.0f}%"
                f"（缺口 {gap:.0f} 个百分点）——Agent 几乎都能跑完，"
                "失分集中在跑完之后的规划与作答，而非流程中断"
            )
        elif completion_rate < _INSIGHT_LOW_COMPLETION_RATE:
            parts.append(f"任务完成率仅 {completion_rate:.1f}%，需优先排查流程贯通问题")

        # Root-cause concentration insight (score_loss is already sorted by total, descending).
        advice = ""
        if total_loss > 0 and len(score_loss) >= _INSIGHT_TOP_ROOT_CAUSES:
            items = list(score_loss.items())
            top1_name, top1_val = items[0]
            top2_name, top2_val = items[1]
            top1_pct = abs(top1_val) / total_loss * 100
            top2_pct = abs(top2_val) / total_loss * 100
            top_sum_pct = top1_pct + top2_pct
            parts.append(
                f"{top1_name}占失分 {top1_pct:.0f}%、{top2_name}占 {top2_pct:.0f}%"
                f"，合计贡献 {top_sum_pct:.0f}% 的分数损失"
            )

            t1 = top1_name.lower()
            if (
                "planning" in t1
                or "strategy" in t1
                or "intent" in t1
                or "规划" in top1_name
                or "策略" in top1_name
                or "意图" in top1_name
            ):
                advice = "提分应优先规划质量与事实校验，不必再投入流程贯通"
            elif (
                "factual" in t1
                or "generation" in t1
                or "completeness" in t1
                or "生成" in top1_name
                or "事实" in top1_name
            ):
                advice = "建议优先提升生成质量与事实准确性"
            elif "tool" in t1 or "工具" in top1_name:
                advice = "建议优先排查工具调用链路的稳定性与数据质量"
        elif total_loss > 0 and len(score_loss) == 1:
            top1_name = next(iter(score_loss))
            parts.append(f"分数损失全部来自 {top1_name}")

        weakest = WebdevReportClient._weakest_dimension_label(spec)
        if weakest:
            parts.append(f"最弱维{weakest}")
        if advice:
            parts.append(advice)

        if not parts:
            return None
        return WebdevReportClient._clip_insight("；".join(parts))

    async def _refine_insight_from_html(self, html: str, spec: dict[str, Any]) -> str:
        """Call AGENT_EVAL_INSIGHT_LLM_* to build today's summary plus top problems; on failure return the input unchanged."""
        fallback = spec.get("insight_text")
        if not isinstance(fallback, str):
            fallback = ""
        has_cases = bool(spec.get("case_results")) or ('id="cases-data"' in html)
        if not fallback.strip() and "insight-banner" not in html and not has_cases:
            if 'id="report-data"' not in html:
                return html
        try:
            raw = await self._complete_insight_llm(
                WebdevReportClient._build_insight_prompt(html, fallback, spec),
            )
        except Exception:
            logger.warning(
                "webdev_client.insight_llm_failed benchmark_name=%s run_id=%s",
                spec.get("benchmark_name"),
                spec.get("run_id"),
                exc_info=True,
            )
            return html
        parsed = WebdevReportClient._normalize_insight_structured(raw)
        if parsed is None:
            logger.warning(
                "webdev_client.insight_llm_empty benchmark_name=%s run_id=%s",
                spec.get("benchmark_name"),
                spec.get("run_id"),
            )
            return html
        replaced = WebdevReportClient._replace_insight_in_html(html, parsed)
        if not WebdevReportClient._insight_rewrite_keeps_case_details(html, replaced):
            logger.warning(
                "webdev_client.insight_rewrite_dropped_cases "
                "benchmark_name=%s run_id=%s — keeping original html",
                spec.get("benchmark_name"),
                spec.get("run_id"),
            )
            return html
        insight = parsed.get("insight_text")
        if isinstance(insight, str) and insight.strip():
            spec["insight_text"] = insight
        summary = parsed.get("insight_summary")
        if isinstance(summary, dict) and summary:
            spec["insight_summary"] = summary
        top_problems = parsed.get("top_problems")
        if isinstance(top_problems, list):
            spec["top_problems"] = top_problems
        logger.info(
            "webdev_client.insight_llm_replaced chars=%d top_problems=%d",
            len(str(insight or "")),
            len(top_problems) if isinstance(top_problems, list) else 0,
        )
        return replaced

    async def _complete_insight_llm(self, prompt: str) -> str:
        """The injected complete wins; otherwise use the Amap gateway primary model plus fallback."""
        if self._insight_llm_complete is not None:
            return await self._insight_llm_complete(prompt)
        return await self._complete_insight_via_gateway(prompt)

    @staticmethod
    def _insight_llm_credentials() -> tuple[str, str, str, str, float] | None:
        """Credentials used only for the report insight. Read the insight config only; do not fall back to judge/example keys."""
        from benchmark.settings.setting import settings  # noqa: PLC0415

        api_key = (settings.insight_llm_api_key or "").strip()
        if not api_key:
            return None
        base_url = (settings.insight_llm_base_url or "").strip().rstrip("/")
        if not base_url:
            return None
        return (
            api_key,
            base_url,
            settings.insight_llm_model,
            settings.insight_llm_model_fallback,
            float(settings.insight_llm_timeout_s),
        )

    async def _complete_insight_via_gateway(self, prompt: str) -> str:
        creds = self._insight_llm_credentials()
        if creds is None:
            return ""
        api_key, base_url, primary, fallback, timeout_s = creds
        for model in (primary, fallback):
            text = await self._post_insight_chat(
                _InsightChatRequest(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    timeout_s=timeout_s,
                ),
            )
            if text:
                return text
            logger.warning("webdev_client.insight_llm_model_empty model=%s", model)
        return ""

    @staticmethod
    async def _post_insight_chat(request: _InsightChatRequest) -> str:
        """OpenAI Chat Completions (same protocol as the ieval Amap gateway). Do not reuse the webdev AsyncClient."""
        from benchmark.utils.http_transport import AsyncClient as InsightClient  # noqa: PLC0415

        url = f"{request.base_url}/chat/completions"
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": _INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": _INSIGHT_LLM_TEMPERATURE,
            "max_tokens": _INSIGHT_LLM_MAX_TOKENS,
        }
        last_error: Exception | None = None
        for attempt in range(_INSIGHT_LLM_RETRIES):
            try:
                async with InsightClient(timeout=request.timeout_s) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {request.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                return content.strip() if isinstance(content, str) else ""
            except Exception as exc:
                last_error = exc
                if attempt < _INSIGHT_LLM_RETRIES - 1:
                    await asyncio.sleep(_INSIGHT_LLM_RETRY_BASE_S * (2**attempt))
        logger.warning(
            "webdev_client.insight_llm_http_failed model=%s error=%s",
            request.model,
            type(last_error).__name__ if last_error else "unknown",
        )
        return ""

    @staticmethod
    def _build_insight_prompt(
        html: str,
        fallback: str,
        spec: dict[str, Any] | None = None,
    ) -> str:
        payload = WebdevReportClient._build_insight_model_payload(spec or {}, html)
        payload_text = json.dumps(payload, ensure_ascii=False)
        if len(payload_text) > _INSIGHT_PAYLOAD_MAX_CHARS:
            payload_text = payload_text[:_INSIGHT_PAYLOAD_MAX_CHARS] + "…[truncated]"
        return _INSIGHT_USER_PROMPT.replace(
            "<<FALLBACK>>",
            fallback or "（无）",
        ).replace("<<PAYLOAD>>", payload_text)

    @staticmethod
    def _build_insight_model_payload(
        spec: dict[str, Any],
        html: str,
    ) -> dict[str, Any]:
        """Assemble the metrics plus case details (including query/output) sent to the insight model."""
        cases = WebdevReportClient._compact_cases_for_insight(spec, html)
        metrics_keys = (
            "run_id",
            "benchmark_name",
            "service_name",
            "dataset_name",
            "report_date",
            "final_score",
            "total_samples",
            "total_success",
            "total_failed",
            "grade_distribution",
            "today_criterion_means",
            "yesterday_criterion_means",
            "criterion_deltas",
            "score_loss_waterfall",
            "attribution_metrics",
            "trend_7d",
        )
        metrics: dict[str, Any] = {}
        for key in metrics_keys:
            val = spec.get(key)
            if val is not None:
                metrics[key] = val
        criteria = spec.get("criteria")
        if isinstance(criteria, list):
            metrics["criteria"] = [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "weight": c.get("weight"),
                }
                for c in criteria
                if isinstance(c, dict)
            ]
        return {
            "metrics": metrics,
            "case_count": len(cases),
            "cases": cases,
        }

    @staticmethod
    def _resolve_insight_raw_cases(spec: dict[str, Any], html: str) -> list[Any]:
        raw_cases = spec.get("case_results")
        if isinstance(raw_cases, list) and raw_cases:
            return raw_cases
        m = _CASES_DATA_PATTERN.search(html)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(1).replace(r"\/", "/"))
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _insight_dim_scores(
        scores: dict[str, Any],
        name_by_id: dict[str, str],
    ) -> dict[str, float]:
        dim_scores: dict[str, float] = {}
        for cid, sc in scores.items():
            if cid == "fused" or not isinstance(sc, dict):
                continue
            sv = sc.get("score")
            if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                label = name_by_id.get(str(cid), str(cid))
                dim_scores[label] = round(float(sv), 3)
        return dim_scores

    @staticmethod
    def _compact_one_case_for_insight(
        case: dict[str, Any],
        name_by_id: dict[str, str],
    ) -> dict[str, Any]:
        case_id = (
            case.get("display_case_id")
            or case.get("case_id")
            or case.get("trace_id")
            or case.get("id")
            or ""
        )
        query = WebdevReportClient._truncate_insight_text(
            case.get("display_input") or case.get("query") or "",
        )
        output = WebdevReportClient._truncate_insight_text(
            case.get("display_output") or case.get("output") or "",
        )
        scores = case.get("scores") if isinstance(case.get("scores"), dict) else {}
        fused = scores.get("fused") if isinstance(scores.get("fused"), dict) else {}
        attr = case.get("attribution")
        if not isinstance(attr, dict):
            attr = fused.get("attribution") if isinstance(fused, dict) else None
        attr_cat = None
        if isinstance(attr, dict):
            attr_cat = attr.get("primary_issue_category") or attr.get("root_cause")
        item: dict[str, Any] = {
            "case_id": str(case_id),
            "query": query,
            "output": output,
            "grade": case.get("grade"),
            "score": fused.get("score"),
            "root_cause": case.get("root_cause_display") or attr_cat,
        }
        dim_scores = WebdevReportClient._insight_dim_scores(scores, name_by_id)
        if dim_scores:
            item["dim_scores"] = dim_scores
        return item

    @staticmethod
    def _compact_cases_for_insight(
        spec: dict[str, Any],
        html: str,
    ) -> list[dict[str, Any]]:
        raw_cases = WebdevReportClient._resolve_insight_raw_cases(spec, html)
        if not raw_cases:
            return []
        criteria = spec.get("criteria") if isinstance(spec.get("criteria"), list) else []
        name_by_id = {
            str(c["id"]): str(c.get("name") or c["id"])
            for c in criteria
            if isinstance(c, dict) and c.get("id")
        }
        out: list[dict[str, Any]] = []
        for case in raw_cases:
            if isinstance(case, dict):
                out.append(WebdevReportClient._compact_one_case_for_insight(case, name_by_id))
        return out

    @staticmethod
    def _truncate_insight_text(value: object, max_chars: int = _INSIGHT_CASE_TEXT_MAX) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    @staticmethod
    def _strip_html_for_insight(
        html: str,
        max_chars: int = _INSIGHT_PAYLOAD_MAX_CHARS,
    ) -> str:
        """Drop the UX patch and keep cases-data, for debugging and compatibility with older callers."""
        stripped = _INSIGHT_PATCH_STYLE_RE.sub("", html)
        stripped = _INSIGHT_PATCH_SCRIPT_RE.sub("", stripped)
        if len(stripped) > max_chars:
            return stripped[:max_chars] + "\n<!-- truncated -->"
        return stripped

    @staticmethod
    def _parse_insight_json_object(raw: str) -> dict[str, Any] | None:
        text = _INSIGHT_FENCE_RE.sub("", raw).strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _normalize_insight_structured(raw: str | None) -> dict[str, Any] | None:
        """Parse model JSON into insight_text / insight_summary / top_problems."""
        if not isinstance(raw, str):
            return None
        obj = WebdevReportClient._parse_insight_json_object(raw)
        if obj is None:
            plain = _INSIGHT_FENCE_RE.sub("", raw).strip().strip("\"'")
            plain = re.sub(r"\s+", " ", plain).strip()
            if len(plain) < _INSIGHT_LLM_MIN_CHARS:
                return None
            return {"insight_text": plain}

        summary = WebdevReportClient._coerce_insight_summary(obj)
        top_problems = WebdevReportClient._coerce_top_problems(obj.get("top_problems"))
        insight_text = obj.get("insight_text") or obj.get("insight") or obj.get("text")
        if not isinstance(insight_text, str) or not insight_text.strip():
            if summary and summary.get("conclusion"):
                insight_text = str(summary["conclusion"])
            else:
                insight_text = ""
        insight_text = re.sub(r"\s+", " ", insight_text.strip())
        if len(insight_text) < _INSIGHT_LLM_MIN_CHARS and not summary and not top_problems:
            return None
        result: dict[str, Any] = {}
        if insight_text:
            result["insight_text"] = insight_text
        if summary:
            result["insight_summary"] = summary
        if top_problems:
            result["top_problems"] = top_problems
        return result or None

    @staticmethod
    def _coerce_insight_summary(obj: dict[str, Any]) -> dict[str, Any] | None:
        raw = obj.get("summary") if isinstance(obj.get("summary"), dict) else None
        if raw is None and isinstance(obj.get("insight_summary"), dict):
            raw = obj.get("insight_summary")
        if not isinstance(raw, dict):
            return None
        conclusion = str(raw.get("conclusion") or "").strip()
        action = str(raw.get("action") or "").strip()
        bullets_raw = raw.get("bullets")
        bullets: list[str] = []
        if isinstance(bullets_raw, list):
            for item in bullets_raw:
                if isinstance(item, str) and item.strip():
                    bullets.append(item.strip())
        if not conclusion and not bullets and not action:
            return None
        out: dict[str, Any] = {}
        if conclusion:
            out["conclusion"] = conclusion
        if bullets:
            out["bullets"] = bullets[:6]
        if action:
            out["action"] = action
        return out

    @staticmethod
    def _sanitize_insight_metric(value: object) -> str:
        """Clean model placeholder copy: insufficient data, unknown, and similar mean it cannot be analyzed, so return an empty string and do not show it."""
        text = str(value or "").strip()
        if not text:
            return ""
        lowered = text.lower().replace(" ", "")
        placeholders = (
            "数据不足",
            "未知",
            "无法分析",
            "无法计算",
            "暂无",
            "不适用",
            "n/a",
            "na",
            "null",
            "none",
            "-",
            "—",
            "–",
        )
        if lowered in placeholders or text in placeholders:
            return ""
        if any(p in text for p in ("数据不足", "无法分析", "无法计算", "未知")):
            return ""
        return text

    @staticmethod
    def _coerce_top_problems(raw: object) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(raw[:_INSIGHT_TOP_PROBLEMS_MAX]):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            chain_raw = item.get("chain")
            chain: list[str] = []
            if isinstance(chain_raw, list):
                chain = [str(x).strip() for x in chain_raw if str(x).strip()]
            rank = item.get("rank")
            if not isinstance(rank, int) or isinstance(rank, bool):
                rank = idx + 1
            problem: dict[str, Any] = {
                "rank": rank,
                "name": name,
                "trend": str(item.get("trend") or "").strip(),
                "affected_cases": item.get("affected_cases"),
                "score_drag": WebdevReportClient._sanitize_insight_metric(
                    item.get("score_drag"),
                ),
                "share_pct": WebdevReportClient._sanitize_insight_metric(
                    item.get("share_pct"),
                ),
                "impacted_dims": str(item.get("impacted_dims") or "").strip(),
                "chain": chain[:6],
                "typical_case_id": str(item.get("typical_case_id") or "").strip(),
                "typical_case_desc": str(item.get("typical_case_desc") or "").strip(),
            }
            out.append(problem)
        return out

    @staticmethod
    def _normalize_insight_output(raw: str | None) -> str | None:
        """Compatibility with older callers: take only the insight_text string."""
        parsed = WebdevReportClient._normalize_insight_structured(raw)
        if not parsed:
            return None
        text = parsed.get("insight_text")
        return text if isinstance(text, str) and text.strip() else None

    @staticmethod
    def _insight_banner_inner(text: str) -> str:
        escaped = _html_escape(text)
        formatted = _INSIGHT_BOLD_RE.sub(r"<strong>\1</strong>", escaped)
        return f"💡 {formatted}"

    @staticmethod
    def _replace_insight_in_html(html: str, insight: str | dict[str, Any]) -> str:
        """Rewrite the report-data insight fields in place; if an old static banner exists, fill insight_text back into it."""
        if isinstance(insight, str):
            payload: dict[str, Any] = {"insight_text": insight}
        else:
            payload = insight
        html = WebdevReportClient._replace_report_data_insight(html, payload)
        insight_text = payload.get("insight_text")
        if isinstance(insight_text, str) and insight_text.strip():
            banner = _INSIGHT_BANNER_RE.search(html)
            if banner:
                inner = WebdevReportClient._insight_banner_inner(insight_text)
                html = (
                    html[: banner.start()]
                    + banner.group(1)
                    + inner
                    + banner.group(3)
                    + html[banner.end() :]
                )
        return html

    @staticmethod
    def _replace_report_data_insight(html: str, insight: str | dict[str, Any]) -> str:
        m = _REPORT_DATA_PATTERN.search(html)
        if not m:
            return html
        try:
            data = json.loads(m.group(1).replace("\\/", "/"))
        except json.JSONDecodeError:
            logger.warning("webdev_client.insight_report_data_invalid")
            return html
        if not isinstance(data, dict):
            return html
        if isinstance(insight, str):
            data["insight_text"] = insight
        else:
            if insight.get("insight_text"):
                data["insight_text"] = insight["insight_text"]
            if insight.get("insight_summary"):
                data["insight_summary"] = insight["insight_summary"]
            if "top_problems" in insight:
                data["top_problems"] = insight["top_problems"]
        fixed_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        tag = '<script type="application/json" id="report-data">' + fixed_json + "</script>"
        return html[: m.start()] + tag + html[m.end() :]

    @staticmethod
    def _insight_rewrite_keeps_case_details(original: str, replaced: str) -> bool:
        """An insight rewrite must not change cases-data, and must not delete an existing CASE detail section."""
        orig_m = _CASES_DATA_PATTERN.search(original)
        new_m = _CASES_DATA_PATTERN.search(replaced)
        if orig_m is not None and (new_m is None or orig_m.group(1) != new_m.group(1)):
            return False
        orig_inspector = _ReportHtmlParser()
        orig_inspector.feed(original)
        if not WebdevReportClient._has_renderable_section_title(
            original, orig_inspector.visible_text, "case 明细"
        ):
            return True
        new_inspector = _ReportHtmlParser()
        new_inspector.feed(replaced)
        return WebdevReportClient._has_renderable_section_title(
            replaced, new_inspector.visible_text, "case 明细"
        )

    @staticmethod
    def _build_template_prompt(spec: dict[str, Any]) -> str:
        """Build a minimal prompt with only structural info, no bulk data.

        Instead of serializing the full ReportSpec JSON (which can exceed
        360K chars with case_results and consume the entire LLM context
        window), this method extracts only the schema/structure info the
        LLM needs to generate the correct HTML template + JS rendering
        logic. The actual data is injected by ``_inject_report_data``
        after the HTML template is returned.

        This keeps the prompt at ~2-5K chars regardless of case_results
        size, giving the LLM the full output token budget for HTML/CSS/JS.
        """
        criteria = spec.get("criteria", [])
        criteria_lines = "\n".join(
            f"  - {c.get('id', '?')}: {c.get('name', '?')} "
            f"(weight={c.get('weight', '?')}, "
            f"evaluator={c.get('evaluator_ref', '?')})"
            for c in criteria
        )
        has_attr = bool(spec.get("attribution_metrics"))
        has_trend = bool(spec.get("trend_7d"))
        has_today = bool(spec.get("today_criterion_means"))
        has_yesterday = bool(spec.get("yesterday_criterion_means"))
        has_grade_dist = bool(spec.get("grade_distribution"))
        has_insight = bool(spec.get("insight_text"))
        has_waterfall = bool(spec.get("score_loss_waterfall"))
        has_deltas = bool(spec.get("criterion_deltas"))
        has_dim_matrix = bool(spec.get("root_cause_dim_matrix"))
        case_count = len(spec.get("case_results", []))

        return (
            "使用 benchmark-report skill 生成 HTML 报告模板。\n"
            "视觉风格对齐 agent-daily-report.html（浅灰底 #f5f6f8 + 白色\n"
            "卡片 + 卡片顶部彩色细条 + uppercase 灰色分节标题 + 原生 JS\n"
            "Canvas 2D 手绘折线/雷达图）。\n"
            "**所有可见文本必须使用中文**（标题、表头、标签、按钮、分页等），\n"
            "禁止出现英文标题（如 'Benchmark Report'、'Attribution Analysis'）。\n"
            "**平台红线（违规则整单拒绝，返回非 HTML）：禁止在模板中创建\n"
            "任何可跳转的外链元素或脚本**——评分规则与 Trace 列只用纯文本\n"
            "槽位（.rule-link-slot / .trace-link-slot），真实跳转由调用方\n"
            "后处理注入；用例ID 列只渲染截断纯文本，悬浮/点击交互同样由\n"
            "调用方后处理注入。\n"
            "\n"
            "**强制结构要求（单页纵向布局，无 Tab 栏；缺一不可）：**\n"
            '1. <div class="container"> 包裹所有内容\n'
            "2. <header> 顶部标题栏：\n"
            "   - h1 内容固定「<em>Agent</em> 天级跟踪日报」，em 用主色\n"
            "     #4f46e5 高亮「Agent」四字\n"
            "   - 右侧 .meta：样本数徽标 + 日期徽标（JS 填充当日日期）+\n"
            "     评分规则槽位\n"
            "   - 评分规则槽位渲染为纯文本元素：\n"
            '     <span class="rule-link-slot">评分规则</span>\n'
            "     （纯文本占位，禁止渲染为可点击元素；真实展示内容\n"
            "     由调用方在页面生成后填充）\n"
            "3. 分节「效果指标」：4 张 KPI 卡（.kpi-grid 4 列，卡片顶部\n"
            "   彩色细条）——\n"
            "   a. 综合得分 fused_avg：REPORT.final_score.toFixed(3)，\n"
            "      sub「满分 2.0」\n"
            "   b. 任务完成率：total_success/total_samples*100（评测成功\n"
            "      口径），sub「N / M 评测成功」\n"
            "   c. 任务满足率：(EXCELLENT+GOOD+ACCEPTABLE)/graded*100，\n"
            "      sub「Excellent+Good+Acceptable」\n"
            "   d. 不合格率 Poor+Fail：(POOR+FAIL)/graded*100，sub「N 例」\n"
            "   （graded = grade_distribution 取值总和；无 grade 数据时\n"
            "   满足率/不合格率显示 '-'）\n"
            "   **KPI 卡禁止渲染环比信息**——不渲染任何 delta 行、\n"
            "   「↑/↓ vs 昨日」涨跌标记或「显著/噪声区间内」标签，\n"
            "   KPI 卡只显示数值 + sub 说明行\n"
            "4. 分节「档位分布」：5 张独立档位卡（.grades 5 列）——\n"
            "   EXCELLENT / GOOD / ACCEPTABLE / POOR / FAIL 各一张卡，\n"
            "   每张卡显示档位名 + 数量 + 占比（分母 = graded）；\n"
            "   **禁止做任何聚合**（不出现优良率/基本合格率/不合格率\n"
            "   聚合卡——那是「效果指标」区的口径）；\n"
            "   按固定顺序渲染，count 为 0 的档位也渲染；\n"
            "   仅当 has_grade_distribution=True 时渲染，False 整栏省略\n"
            "5. 分节「近 7 天趋势」（仅当 has_trend_7d=True 时渲染，\n"
            "   False 整栏省略）：一张卡片，Canvas 2D 手绘双轴**折线图**\n"
            "   （**仅折线 + 数据点，禁止面积填充/柱状图/堆积图**；\n"
            "   createElement('canvas') + getContext('2d')，缓冲区按\n"
            "   clientWidth×260×devicePixelRatio 设置并用 setTransform 缩放\n"
            "   保证高清）——左轴「综合得分」（0-2，实线 #4f46e5，数据\n"
            "   trend_7d.fused_avg），右轴「优良率%」（虚线\n"
            "   setLineDash([5,3]) #16a34a，数据\n"
            "   trend_7d.excellent_good_rate），x 轴按 dates 均分；\n"
            "   null 点跳过（分段 beginPath/moveTo/lineTo/stroke）；\n"
            "   网格横线 + 轴刻度文字 + x 轴 MM-DD 标签（ctx.fillText，\n"
            "   11px #6b7194）；数据点画实心圆（arc + 白边）并记录坐标，\n"
            "   canvas mousemove 命中检测（最近点 8px 内）显示浮动 div\n"
            "   提示「MM-DD：数值」，mouseleave 隐藏；**禁止任何图表库\n"
            "   存在性检查**——页面不引入图表库，`typeof Chart` / `new Chart(`\n"
            "   / 「图表库未加载」分支一律禁止出现（search_eval-20260916：\n"
            "   原生 2D 趋势图残留 Chart.js 守卫导致整节空白）\n"
            "6. 分节「评测维度」：单卡紧凑布局（一张卡片，卡内 flex\n"
            "   横排：左表右雷达；雷达区 flex 居中容器，canvas 高度约\n"
            "   240px，保证一眼看全所有维度）——\n"
            "   a. 左「维度得分明细」表格（列：维度/权重/得分/满分率/\n"
            "      分布）：行 = REPORT.criteria 顺序；得分 =\n"
            "      criterion_metrics[id].mean.toFixed(3)；**满分按维度区分**\n"
            "      ——criterion id 以 d0/D0 开头或名称含「安全」的一票否决\n"
            "      维度满分 1 分（0-1 分制），满分率 = mean/1*100%，\n"
            "      分布条颜色阈值同比缩放（均值≥0.8 绿 / ≥0.6 琥珀 / 红）；\n"
            "      其余维度满分 2 分（0-2 分制），满分率 = mean/2*100%，\n"
            "      颜色：均值≥1.6 绿 / ≥1.2 琥珀 / 红；满分率均保留 1 位\n"
            "      小数；分布 = 水平进度条（宽度 = 满分率）；\n"
            "      权重列：一票否决维度行显示红色小标「一票否决」，\n"
            "      其余显示灰底权重百分比；**维度列按内容收窄**\n"
            "      （white-space:nowrap; width:1%），与权重列间距不超过\n"
            "      3 个字符（padding-right:3ch），分布列吃剩余宽度，\n"
            "      保证一眼看全所有维度行\n"
            "   b. 右侧居中「维度雷达（今日 vs 昨日）」Canvas 2D 手绘雷达图\n"
            "      （同样 createElement('canvas') + getContext('2d')，\n"
            "      devicePixelRatio 高清缓冲）：labels = 各 criterion name\n"
            "      **加满分后缀**——格式「维度名 /N」（N = 该维度满分，如\n"
            "      「安全与合规 /1」、「意图识别 /2」；维度名超 8 字先截断\n"
            "      加省略号再拼后缀），ctx.fillText 11px #6b7194，\n"
            "      textAlign 按左右半侧 start/end——满分必须随轴标签直接\n"
            "      可见，禁止只依赖悬停 tooltip；\n"
            "      **按维满分归一化（方案 A）**：各轴几何半径占比\n"
            "      v = clamp(mean / full_score, 0, 1)，其中 full_score =\n"
            "      一票否决维（id 以 d0/D0 开头或名称含「安全」）为 1，\n"
            "      其余为 2——**任一维满分时折线顶点到达最外圈**，\n"
            "      禁止再用 mean/2 的绝对分定位把满分 1 的维压在半半径；\n"
            "      网格四层为满分占比环 25%/50%/75%/100%（rv=0.25/0.5/\n"
            "      0.75/1.0）——**不画任何刻度文字**（禁止「25%」「50%」\n"
            "      「75%」「100%」等百分比刻度标签，网格环仅作视觉参考，\n"
            "      数值读数由轴标签满分后缀与悬停 tooltip 承担）；\n"
            "      各轴画中心→外圈实线 #e8e9ee（一票否决轴与普通轴同\n"
            "      样画满，**禁止**半半径短横档封顶 / 外段虚线等旧绝对分补丁）；\n"
            "      顶点样式——今日点 #4f46e5 radius 3.5；昨日点 #d97706\n"
            "      radius 3.5；当 mean/full_score≥0.95 时该系列顶点可略\n"
            "      放大至 radius 5（仍用该系列颜色），**禁止**在图上绘制\n"
            "      「满分」字样（含 fillText「满分」、旁侧标注、图例外\n"
            "      任何「满分」标签）；\n"
            "      今日值 = today_criterion_means[id]、昨日值 =\n"
            "      yesterday_criterion_means[id]——两者同为\n"
            "      {criterion_id: 数值} 扁平 map，用同一个取值函数直接\n"
            "      取数值（**禁止**访问 criterion_metrics 取雷达值——其\n"
            "      值是 {mean, ...} 嵌套对象，直接相除得 NaN 会导致雷达\n"
            "      空白）；绘图前再除以该维 full_score；\n"
            "      N 边形顶点 angle_i = (-90 + i*360/N) 度（i=0 即正上，\n"
            "      criteria[0] 通常为安全与合规），\n"
            "      x = cx + R*v*cos(angle)、y = cy + R*v*sin(angle)；\n"
            "      网格画四层同心多边形 + 中心到各顶点轴线；**雷达只画线\n"
            "      不对面积染色**（closePath 后只 stroke，**禁止 fill**）；\n"
            "      **数据系列多边形必须闭环**——逐维度 moveTo/lineTo 后必须\n"
            "      ctx.closePath() 把最后一个顶点连回起点（或循环 i<=N 用\n"
            "      angles[i%N] 回到首点），否则雷达图最后一条边缺口不闭环；\n"
            "      **系列线型必须明显可区分**：今日 = 实线 #4f46e5\n"
            "      lineWidth 2.5、setLineDash([])；昨日有数据时 =\n"
            "      虚线 setLineDash([8,4]) #d97706 lineWidth 2.5；\n"
            "      **可点选图例（强制）**：在 canvas **下方**放图例条\n"
            "      （flex 居中，gap 16px），每项含「线段样例 + 文字」——\n"
            "      今日样例用 28×3px 实线色条 #4f46e5，昨日样例用\n"
            "      28×3px 虚线色条（border 或 background 重复渐变模拟\n"
            "      dash，色 #d97706），文字「今日」「昨日」；cursor:pointer；\n"
            "      默认两系列均选中（有昨日数据时）；点击某项切换选中，\n"
            "      未选中项 opacity 0.35 + 文字删除线；**重绘时雷达只 "
            "stroke 当前选中系列**（只选今日则不画昨日，只选昨日则\n"
            "      不画今日，都选则叠加，都未选则只保留网格与轴标签）；\n"
            "      无昨日数据时图例仅「今日」一项且不可关；\n"
            "      **悬停必须显示得分**——仅对**当前可见系列**的顶点做\n"
            "      命中检测（最近点 12px 内）显示浮动 div 提示\n"
            "      「维度名：原始 mean / 满分」（如「意图理解：1.80 / 2」），\n"
            "      mouseleave 隐藏；**tooltip div 必须写在 radar-box\n"
            "      （canvas 容器）内部作为其子元素**——JS 用\n"
            "      box.insertBefore(canvas, tooltip) 挂载 canvas，tooltip\n"
            "      在 box 外会抛 NotFoundError 导致雷达图整体空白；\n"
            "      insertBefore 的参照节点必须是该挂载容器自身的子元素\n"
            "      （不能是兄弟节点或 createElement 后未入 DOM 的游离\n"
            "      节点）；tooltip 绝对定位、radar-box position: relative\n"
            "      **单画布重绘**：renderRadar 每次执行必须先移除 radar-box\n"
            "      内所有旧 canvas 再挂载新 canvas（init/图例切换/resize 都会\n"
            "      触发重绘，禁止叠加多画布）\n"
            "7. 分节「不合格归因」（仅当 has_attribution_metrics=True 时\n"
            "   渲染，False 整栏省略）：标题「不合格归因」+ total_attributed\n"
            "   总数徽标，渲染规则见下方「不合格归因统计区域」\n"
            "8. Case 明细：.table-wrapper（overflow: auto 双向滚动容器）+\n"
            "   sticky thead + .detail-modal\n"
            "9. CSS 调色板（与 agent-daily-report.html 对齐）：主色\n"
            "   #4f46e5，绿 #16a34a，红 #dc2626，琥珀 #d97706，蓝 #2563eb，\n"
            "   紫 #7c3aed，青 #0891b2，墨绿 #0d9488；页面底 #f5f6f8、\n"
            "   卡片白底 + 边框 #e2e4ea + 柔和阴影；分节标题 .sec 风格\n"
            "   （小号灰色大写字距）\n"
            "10. **综合得分必须显示数值**——REPORT.final_score 是浮点数，\n"
            "    直接用 .toFixed(3) 格式化输出，不要用条件判断返回 '-'。\n"
            "    如果 final_score 为 null 才显示 '-'。\n"
            "\n"
            "**重要：不要在 HTML 中硬编码任何数据值。**\n"
            "报告数据将通过 JSON <script> 标签由调用方注入，\n"
            "你的 JavaScript 代码负责从这些标签读取数据并动态渲染。\n"
            "\n"
            "报告结构信息（用于生成正确的 HTML 结构和 JS 渲染逻辑）：\n"
            f"- benchmark_name: {spec.get('benchmark_name', '?')}\n"
            f"- service_name: {spec.get('service_name', '?')}\n"
            f"- dataset_name: {spec.get('dataset_name', '?')}\n"
            f"- run_id: {spec.get('run_id', '?')}\n"
            f"- final_score: {spec.get('final_score', '?')}\n"
            f"- total_samples: {spec.get('total_samples', '?')}\n"
            f"- aggregation_method: {spec.get('aggregation_method', '?')}\n"
            f"- criteria ({len(criteria)}):\n"
            f"{criteria_lines}\n"
            f"- has_attribution_metrics: {has_attr}\n"
            f"- has_trend_7d: {has_trend}\n"
            f"- has_today_criterion_means: {has_today}\n"
            f"- has_yesterday_criterion_means: {has_yesterday}\n"
            f"- has_grade_distribution: {has_grade_dist}\n"
            f"- has_insight_text: {has_insight}\n"
            f"- has_score_loss_waterfall: {has_waterfall}\n"
            f"- has_criterion_deltas: {has_deltas}\n"
            f"- has_root_cause_dim_matrix: {has_dim_matrix}\n"
            f"- has_case_results: {case_count > 0} (count={case_count})\n"
            "\n"
            "**has_* 标志是条件分节的唯一判断依据**——真实数据由调用方\n"
            "注入，你看不到 REPORT 内容：has_xxx=True 时对应分节必须渲染，\n"
            "False 时才允许整栏省略。\n"
            "\n"
            "你的 HTML 必须包含以下两个 JSON script 标签"
            "（内容用占位符，调用方会注入真实数据）：\n"
            '<script type="application/json" id="report-data">{}</script>\n'
            '<script type="application/json" id="cases-data">[]</script>\n'
            "\n"
            "你的 JS 必须从这些标签读取数据并渲染所有内容：\n"
            "var REPORT = JSON.parse("
            "document.getElementById('report-data').textContent);\n"
            "var CASES = JSON.parse("
            "document.getElementById('cases-data').textContent);\n"
            "\n"
            "report-data 包含完整 ReportSpec JSON（不含 case_results）：\n"
            "- benchmark_name, service_name, dataset_name, run_id,\n"
            "  final_score, aggregation_method, total_samples,\n"
            "  total_failed, total_success\n"
            "  （总结区域必须用 total_failed/total_success 显示失败/成功数，\n"
            "  它们是 item 级别去重计数；criterion_metrics 中不再有 failed_count 字段。\n"
            "  **禁止**从 criterion_metrics 的 sample_count - success_count 计算每维度\n"
            "  失败数再求和——一个 case 在 3 个维度都失败应计为 1 个失败 case，\n"
            "  不是 3 个。总结卡片失败数 = REPORT.total_failed，不可用其他方式计算。）\n"
            "- criteria: [{id, name, description, weight, evaluator_ref}]\n"
            "- criterion_metrics: {criterion_id: {mean, median, p95,\n"
            "  min, max, std, pass_rate, sample_count, success_count,\n"
            "  evaluator_ref, criterion_id}}（嵌套统计对象，仅供维度\n"
            "  得分明细表使用；雷达图禁止读取本字段）\n"
            "- attribution_metrics: {key: {field_distributions:\n"
            "  {字段名: {取值: count}}, value_labels:\n"
            "  {字段名: {取值: 展示标签}}, total_attributed}}\n"
            "  （字段名不固定——由数据决定，当前如 primary_issue_category，\n"
            "  其他业务可能是 root_cause / severity 等，全部字段都要渲染；\n"
            "  value_labels 是取值的展示标签（如 tool/tool_timeout），\n"
            "  仅展示层替代原取值，统计数量/排序不变，可能缺失）\n"
            "- grade_distribution: {EXCELLENT: n, GOOD: n, ACCEPTABLE: n,\n"
            "  POOR: n, FAIL: n}（rubric_v2 等级计数，由调用方从\n"
            "  case_results[].grade 预计算；缺失/为空时档位分布栏省略）\n"
            "- trend_7d: {dates: [MM-DD...], fused_avg: [...],\n"
            "  excellent_good_rate: [...], poor_fail_rate: [...],\n"
            "  eval_success_rate: [...], task_satisfaction_rate: [...]}\n"
            "  （近 7 天趋势序列：各数组等长、按日期升序、缺失点为\n"
            "  null、最后一位是今日；由调用方装配，缺失时「近 7 天\n"
            "  趋势」栏省略；趋势数据仅用于趋势折线图，**KPI 卡\n"
            "  不渲染任何 delta/环比行**；task_satisfaction_rate\n"
            "  不上趋势图）\n"
            "- today_criterion_means: {criterion_id: mean}（今日各维度\n"
            "  均值扁平 map，雷达图「今日」数据集唯一数据源）\n"
            "- yesterday_criterion_means: {criterion_id: mean}（昨日各\n"
            "  维度均值，与 today_criterion_means 同构；缺失时雷达只画\n"
            "  今日数据集）\n"
            "- insight_text: 自动洞察文本（由调用方基于完成率 + root_cause_display\n"
            "  的 fused 失分**合计**生成中文文案；缺失时不渲染洞察横幅）\n"
            "- score_loss_waterfall: {start, end, delta, steps:[{label,loss,count}]}\n"
            "  （分数损失瀑布；has_score_loss_waterfall=True 时必须渲染）\n"
            "- criterion_deltas: [{id,name,today,yesterday,delta}]\n"
            "  （维度环比；has_criterion_deltas=True 时必须渲染；KPI 卡仍禁止环比行）\n"
            "- root_cause_dim_matrix: {rows, cols, cells, unit}\n"
            "  （根因×维度平均失分热力；has_root_cause_dim_matrix=True 时必须渲染）\n"
            "- **增强分析布局**：分数损失归因与维度环比必须位于同一个\n"
            "  #waterfall-delta-section 内，桌面端同一行左右两列且等宽；\n"
            "  仅在 ≤900px 窄屏下改为上下排列，禁止生成两个全宽分节。\n"
            "- **增强分析数值精度**：分数损失与维度环比中的 start / end /\n"
            "  steps[].loss / today / yesterday / delta，以及根因×维度失分矩阵\n"
            "  的有效单元格，全部用 Number(value).toFixed(3) 固定显示 3 位\n"
            "  小数（包括 0 → 0.000）；仅缺失/null/非数值显示「—」。\n"
            "- **增强分析计算说明（必须紧贴对应栏底部渲染）**：\n"
            "  1. 分数损失归因：总落差=max(0,满分−综合分)；按各根因\n"
            "     （含“未归因”）有效 case 的 fused 失分合计占比分摊。\n"
            "  2. 维度环比：今日同维度有效样本均分−昨日同 benchmark\n"
            "     已完成 run 的同维度有效样本均分；绝对变化≤0.010 按持平\n"
            "     显示，其余正值提升、负值下降。\n"
            "  3. 根因×维度矩阵：该根因（含“未归因”）下所有含该维有效\n"
            "     分数 case 的（维度满分−得分）均值；满分 case 按0计入。\n"
            "  三段说明使用 muted 小字，禁止合并到章节外或省略。\n"
            "\n"
            "cases-data 包含 case_results 数组：\n"
            "- 每条: {case_id, display_case_id, trace_id?, input, output,\n"
            "  display_input?, display_output?, expected?, pass, grade?,\n"
            "  trace_url?, root_cause_display?, scores: {cid: {score,\n"
            "  reason, eval_failed, judge?, attribution?}}, sub_scores?,\n"
            "  attribution?}\n"
            "  （pass 是布尔值——有归因数据的 case 永远为 false；\n"
            "  状态列不使用 pass，只用 grade 五等级；\n"
            "  grade 是 rubric_v2 等级（EXCELLENT/GOOD/ACCEPTABLE/POOR/FAIL，\n"
            "  由 scores.fused.score 预计算），case 状态列只用它，\n"
            "  仅 fused 分有效且评测成功的 case 才有；\n"
            "  trace_id 是 judge 回传的数据源原始 trace 标识（外部回调\n"
            "  写入 fused 行，可能缺失），展示在 query 下拉「基本信息」\n"
            "  的用例ID 行（表格不再单独开「用例ID」列）；\n"
            "  trace_url 是 trace 关联保留字段（外部回调写入 fused\n"
            "  行，可能缺失），由调用方后处理使用，你的 JS 无需读取；\n"
            "  scores.fused.score 是写入 run_items 的实际综合分数，\n"
            "  综合分列直接取此值，不要自行计算均值；\n"
            "  display_case_id 是业务 task_id（fallback case_id），\n"
            "  详情下拉/modal 的「用例ID」行显示它；\n"
            "  display_input 在详情 modal 用「query」行展示：默认截断\n"
            "  （约 80 字 + …），hover 浮层可选择复制全文，禁止把完整\n"
            "  query 直接铺满 modal；\n"
            "  display_input 是调用方从 input 提取的用户 query\n"
            "  （顶层 query 或 request_body.query），表格「query」列用它，\n"
            "  不要在前端解析 input）\n"
            "- **Case 明细表格列白名单（只允许这 7 类，从左到右）**：\n"
            "  1. query：display_input（调用方预提取的用户 query 文本），\n"
            "     单元格截断（约 40 字 + …），**点击展开行内下拉**\n"
            "     （.trace-detail-row：基本信息→输入→输出→各维度评分→\n"
            "     用例归因）；hover 浮层展示全文且 user-select:text 可复制\n"
            "     （禁止只用 title）；空显示 muted '-'；禁止前端解析 input\n"
            "     JSON 再提取（直接用 display_input 字段）\n"
            "  2. output：display_output（调用方预提取的模型输出 result\n"
            "     文本），单元格截断（约 40 字 + …，**允许多行换行**——\n"
            "     white-space:normal + overflow-wrap:anywhere 自动换行，\n"
            "     禁止 nowrap 单行省略）；**列宽固定 10em（每行约 10 个\n"
            "     汉字，table-layout:auto 下禁止被其他 nowrap 列挤压成\n"
            "     每行仅 1-2 字的窄列）**；hover 浮层展示全文；空显示\n"
            "     muted '-'；禁止前端解析 output JSON（直接用\n"
            "     display_output 字段）\n"
            "  3. case状态：grade 五等级徽标（EXCELLENT/GOOD 绿色、\n"
            "     ACCEPTABLE 蓝色、POOR/FAIL 红色）；grade 缺失显示\n"
            "     muted '-'——**状态只有这五个等级，禁止「通过/不通过」\n"
            "     徽标**（pass 字段不用于状态列渲染）\n"
            "  4. 综合分：scores.fused.score\n"
            "  5. 各维度得分：**单列合并展示**（禁止每个 criterion 一列），\n"
            "     列名「各维度得分」，**每个 criterion 独占一行**\n"
            "     （纵向堆叠，自上而下按 criteria 顺序排列），\n"
            "     格式「维度名：得分」；**实现红线**：每个维度用独立\n"
            "     block 级元素渲染（每维度一个 <div>）——禁止用\n"
            "     空格/逗号/顿号分隔把各维度横排在同一行，禁止仅依赖\n"
            "     换行符 + white-space:pre-line 实现换行（表格 td 的\n"
            "     white-space:nowrap 规则特异性更高会覆盖它，导致\n"
            "     各维度横置挤在一行），该单元格自身也不得设置\n"
            "     white-space:nowrap；\n"
            "     **只渲染 judge 得分**——该列禁止出现任何归因内容\n"
            "     （attribution 在 query 下拉「用例归因」栏展示）\n"
            "  6. 根因：root_cause_display（调用方预计算，缺失显示\n"
            "     muted '-'），#eef2ff 底 #4338ca 字的小号 badge\n"
            "  7. Trace链接：单元格渲染为纯文本槽位\n"
            '     <span class="trace-link-slot">-</span>（纯占位文本元素，\n'
            "     无交互）；该列真实展示内容由调用方在页面生成后填充，\n"
            "     你的 JS 不需要读取 trace_url，也不要给该列添加任何\n"
            "     交互行为\n"
            "  **禁止生成「用例ID」「输入」「操作」列**——用例ID 写在\n"
            "  query 下拉「基本信息」；完整 input/output 只在下拉中展示；\n"
            "  表格首列是「query」；query / output 列只展示 display_input /\n"
            "  display_output 短文本。\n"
            '  **Case 明细 <table> 必须带 id="data-table"**（tbody 可另设\n'
            '  id="case-tbody"，但 table 本身禁止只用 class="case-tbl"\n'
            "  而漏 id——调用方后处理按 #data-table 定位；漏 id 会导致\n"
            "  query 下拉/悬浮失效、Trace链接列停在占位 '-'）。\n"
            "  **query / case状态 / 综合分 / 各维度得分 / Trace链接 为\n"
            "  必有值列**——调用方已对每条 case 做必填校验（缺失记\n"
            "  warning 日志）；渲染仍按缺失兜底 muted '-'，不得因单条\n"
            "  数据缺失中断整表渲染。\n"
            "\n"
            "attribution 字段结构（Aone 归因回调，字段不固定出现）：\n"
            "- attribution.primary_issue_layer: 问题层（tool/planning/generation/none）\n"
            "- attribution.primary_issue_category: 问题分类\n"
            "- attribution.primary_issue_detail: 归因推理过程\n"
            "- attribution.severity: 严重程度（critical/major/minor）\n"
            "- attribution.root_cause_chain: 根因链\n"
            "- attribution.improvement_suggestion: 改进建议\n"
            "- attribution.confidence: 置信度 0.0-1.0\n"
            "- attribution.is_eval_misjudge: 是否评测误判\n"
            "- attribution.triage_level: 分级标签（兼容旧字段）\n"
            "- attribution.root_cause: 根因（兼容旧字段）\n"
            "- attribution.failure_modes: 失败模式列表\n"
            "- attribution.summary: 归因摘要\n"
            "  字段可能不全部出现，有什么渲染什么。\n"
            "- root_cause_display: 「根因」列显示文本（调用方预计算，按业务分支）：\n"
            "  - ieval（benchmark_name 以 ieval 开头）：归因数据中的\n"
            "    primary_issue_layer / primary_issue_category（经 value_labels\n"
            "    转展示标签 + 去重前缀 + **转中文**，与「综合归因」统计表格对齐，\n"
            "    如 planning/strategy_error → 规划/策略错误）；无归因时回退\n"
            "    scores[cid] 未满分维度中文名。\n"
            "  - search_eval：judge 中 sub_scores 所有未满分维度的中文标识\n"
            "    （criteria 名，不读归因）。\n"
            "  - 其余业务：归因主因 root_cause 原文；无归因（或无主因）时回退\n"
            "    未满分维度中文名（优先 scores[cid]，回退 sub_scores）。\n"
            "  已去重前缀（layer 与 category label 的重复前缀只保留更具体的一段）；\n"
            "  primary_issue_layer=none 视为无有效归因，回退未满分维度；\n"
            "  可能缺失；缺失时根因列显示 muted「-」\n"
            "\n"
            "渲染要求：\n"
            "1. **单页纵向布局，无 Tab 栏**——不要生成 .tabs-container、\n"
            "   tab 按钮或 tab-content 切换；分节从上到下依次：Banner\n"
            "   （AI Agent迭代质量）→ header → 自动洞察横幅 → 效果指标 →\n"
            "   档位分布 → 近 7 天趋势 → 评测维度 → 分数损失与维度环比 →\n"
            "   根因×维度失分矩阵 → 不合格归因 → Case 明细；每节标题用\n"
            "   .sec 风格分节标题。（条件分节：对应 has_*=False 时整栏省略；\n"
            "   has_score_loss_waterfall / has_criterion_deltas /\n"
            "   has_root_cause_dim_matrix 为 True 时必须渲染对应卡；\n"
            "   **禁止**渲染废止字段 score_loss_attribution / score_loss_matrix）\n"
            "2. **图表纯原生实现，零外部依赖**（趋势图与雷达图）：\n"
            "   **禁止引入任何图表库/CDN/外部 <script src>**，\n"
            "   且**禁止使用 SVG**——部分内嵌查看器会\n"
            "   剥离内联 SVG 元素，导致数据文字可见但图表空白：用原生\n"
            "   JS + Canvas 2D 手绘——JS 计算顶点/坐标后在 canvas 上绘制\n"
            "   （document.createElement('canvas') + getContext('2d')，\n"
            "   缓冲区 = clientWidth×260×devicePixelRatio，ctx.setTransform\n"
            "   缩放，window resize 防抖 ~150ms 重绘）；文字色 #6b7194、\n"
            "   ctx.font=\"11px -apple-system,'SF Pro Text',sans-serif\"；\n"
            "   图表容器固定高度（如 .chart-box{height:260px}）+ width:100%\n"
            "   自适应。\n"
            "3. **KPI 卡禁止渲染环比信息**——不渲染任何 delta 行、\n"
            "   「↑/↓ vs 昨日」涨跌标记、「显著/噪声区间内」标签或\n"
            "   任何与昨日的对比数据；REPORT.trend_7d 仅用于「近 7 天\n"
            "   趋势」折线图，KPI 卡只显示数值 + sub 说明行。\n"
            "4. **不合格归因统计区域**（章节标题「不合格归因」，右侧\n"
            "   total_attributed 总数徽标「N 例」；不要用英文标题）：\n"
            "   对 attribution_metrics 每个 key（fused →\n"
            "   「综合归因」，criterion id → criterion name），遍历其\n"
            "   field_distributions 中的每个字段名，各渲染一个分布块：\n"
            "   字段名为标题，按 count 降序展示每个取值的个数和占比，\n"
            "   并显示 total_attributed 总数。\n"
            "   不要硬编码字段名——数据里有什么字段就渲染什么字段。\n"
            "   条形图行的标签：若该 key 的 value_labels[字段名][取值] 存在，\n"
            "   用该展示标签（如 tool/tool_timeout）替代原始取值；\n"
            "   数量、占比、排序仍按 field_distributions，不变。\n"
            "   **「综合归因」（fused）的 primary_issue_category 字段必须\n"
            "   渲染为表格**（而非横向条形图）：四列——分类 / 分布 / 个数 /\n"
            "   百分比；分类列用中文展示标签（已知分类的英文标签需转中文：\n"
            "   planning/strategy_error→规划/策略错误、planning/intent_error→\n"
            "   规划/意图错误、planning/context_lost→规划/上下文丢失、\n"
            "   tool/tool_exec_error→工具/执行错误、tool/tool_unavailable→\n"
            "   工具/不可用、tool/data_insufficient→工具/数据不足、\n"
            "   generation/factuality_error→生成/事实性错误、\n"
            "   generation/completeness_error→生成/完整性错误；未知标签保留\n"
            "   原文）；分布列为水平进度条（宽度 = 该分类 count / 最大 count，\n"
            "   进度条宽度约占列宽 42%，高度 10px，圆角 5px，主色 #4f46e5）；\n"
            "   个数列右对齐；百分比列右对齐保留 1 位小数。其余字段仍按横向\n"
            "   条形图渲染。\n"
            "5. case 明细中每条 case 的状态：\n"
            "   只使用 case_results[].grade（rubric_v2 等级）渲染徽标，\n"
            "   徽标文字直接用等级词：POOR、FAIL → 红色 #dc2626；\n"
            "   EXCELLENT、GOOD → 绿色 #16a34a；ACCEPTABLE → 蓝色\n"
            "   #2563eb（冷色，必须与 POOR/FAIL 红色明显区分，禁用\n"
            "   琥珀/橙红色）。\n"
            "   grade 缺失（评测失败等）时显示 muted '-'——禁止使用\n"
            "   「通过/不通过」徽标（pass 不用于状态列）。\n"
            "   **有归因数据的 case grade 只会是 POOR/FAIL/ACCEPTABLE**，\n"
            "   不要根据分数颜色推断等级。详情 modal 综合分旁同步展示等级。\n"
            "6. **表格行禁止渲染归因内容**——「各维度得分」等列只放\n"
            "   judge 得分；attribution 只在「详情」modal 的「用例归因」\n"
            "   区展示（渲染所有出现的字段，至少包含 primary_issue_layer、\n"
            "   primary_issue_detail、severity、root_cause_chain、\n"
            "   improvement_suggestion、confidence）；\n"
            "   primary_issue_category 仅用于不合格归因统计分布图，\n"
            "   任何 case 级展示都不渲染它\n"
            "7. Case 明细筛选与分页：\n"
            "   - 筛选控件只有两个：搜索框（精确匹配 用例ID：trace_id /\n"
            "     case_id / display_case_id，或 query：display_input；\n"
            "     trim 后全等，空串不过滤；placeholder「精确搜索用例ID / query」）+\n"
            "     状态下拉（**默认选中「不合格(FAIL+POOR)」**，选项顺序：\n"
            "     全部状态 / 不合格(FAIL+POOR) / EXCELLENT / GOOD / ACCEPTABLE /\n"
            '     POOR / FAIL；「全部状态」option 的 value 必须为空串（""）；\n'
            "     选中「不合格(FAIL+POOR)」时只展示 grade 为 FAIL\n"
            "     或 POOR 的 case，其余档位只在选择对应项时显示）；\n"
            "     **禁止生成维度筛选下拉**（不得出现「全部维度」select，不提供\n"
            "     按 criterion 过滤 case 的下拉）；\n"
            "   - **筛选作用于全量 CASES 数组**（全局筛选）——先过滤\n"
            "     全量数据，再对过滤结果分页（禁止只筛当前页）；\n"
            "     筛选变化重置到第 1 页；\n"
            "   - 分页每页 10 条：renderTable 每次用 filtered.slice(\n"
            "     (page-1)*10, page*10) 完整重建 tbody——翻页/筛选后\n"
            "     每行都在，禁止丢数据；\n"
            "   - 分页控件显示「共 N 条」（N = 过滤后条数）\n"
            "8. **表格容器双向滚动 + 鼠标拖拽平移**：.table-wrapper 设\n"
            "   overflow: auto; max-height: 70vh——表格体超高纵向滚动，\n"
            "   超宽横向滚动（单元格 white-space: nowrap 自然展开，\n"
            "   禁止压缩列宽/折行截断内容，靠滚动查看完整内容；\n"
            "   **query 列例外**：截断 + hover 浮层复制全文）；\n"
            "   sticky thead（top: 0）滚动时表头常驻；支持鼠标按住\n"
            "   拖动平移（mousedown 后拖动改变 scrollLeft/scrollTop，\n"
            "   cursor: grab / grabbing；拖动位移>5px 时不触发行点击\n"
            "   与行点击），滚轮/触控板滚动保持原生行为\n"
            "9. **综合得分卡片**必须显示 REPORT.final_score 的数值，\n"
            "   用 .toFixed(3) 格式化。不要做任何条件判断——\n"
            "   如果 REPORT.final_score 不为 null 就直接显示。\n"
            "10. **init/主流程中调用的函数必须先定义且命名完全一致**\n"
            "    （禁止调用 renderXxx 却定义 drawXxx）；init 各渲染\n"
            "    步骤之间互不依赖——**每步调用必须用独立 try/catch 包裹**\n"
            "    （catch 内 console.error 后继续），单步失败不得中断后续步骤。\n"
            "11. **条件分节（近 7 天趋势 / 不合格归因）禁止只渲染\n"
            "    display:none 占位**——若容器初始隐藏，JS 必须在数据\n"
            "    非空时显式解除隐藏（style.display=''），否则该节永远\n"
            "    不可见。\n"
            "12. **自动洞察横幅**（仅当 has_insight_text=True 时渲染，\n"
            "    False 整栏省略）：header 下方、效果指标上方，黄色底\n"
            "    （#fefce8）+ 左侧边框 3px #eab308 的横幅卡片，内含\n"
            "    灯泡 emoji 💡 + 文本内容。文本中 **加粗** 部分用\n"
            "    <strong> 标签包裹（如完成率、综合分、根因名称）。\n"
            "    样式：padding 12px 16px，border-radius 8px，\n"
            "    font-size 13px，line-height 1.6，color #713f12。\n"
            "13. **根因分布点击联动过滤**：不合格归因的 distribution bar\n"
            "    行（.dist-row）加 cursor: pointer + hover 高亮\n"
            "    （background #f0f1f4）；点击某行时，将该根因值写入\n"
            "    CASE 明细的筛选条件（新增根因筛选维度），触发\n"
            "    applyFilters() 重新渲染表格；再次点击同一行取消筛选。\n"
            "    当前选中的根因行加左侧 3px 主色边框标识。\n"
            "14. **CASE 明细表格「根因」列**（第 6 列，在 Trace链接\n"
            "    之前）：**直接渲染 case.root_cause_display**（调用方\n"
            "    预计算的根因显示文本——归因主因 → 问题层/分类 →\n"
            "    未满分维度中文名；禁止 JS 从 attribution/scores 自行\n"
            "    推导）；标签样式为小号 badge（padding 2px 6px，\n"
            "    border-radius 4px，font-size 10px，背景 #eef2ff，\n"
            "    文字色 #4338ca）；字段缺失时显示 muted '-'。\n"
            "15. **CASE 明细「各维度得分」列加迷你条形图**：在每个维度\n"
            "    得分文字右侧加水平迷你进度条（宽度 40px，高度 4px，\n"
            "    圆角 2px）；条的宽度 = 得分/满分*100%，颜色按得分\n"
            "    阈值（其余维度 ≥1.6 绿 / ≥1.2 琥珀 / 红；D0/安全维度\n"
            "    同比缩放 ≥0.8 绿 / ≥0.6 琥珀 / 红）；仅显示失分项\n"
            "    （得分 < 满分 80%）的条形图加琥珀/红色，满分项不渲染\n"
            "    条形图（减少视觉噪音）。\n"
            "16. **档位分布卡片加顶部彩色条**：每张档位卡（.grades .gc）\n"
            "    顶部加 3px 彩色细条——EXCELLENT/GOOD 用 #16a34a，\n"
            "    ACCEPTABLE 用 #2563eb，POOR/FAIL 用 #dc2626；\n"
            "    通过 CSS ::after 伪元素实现（与 KPI 卡一致）。\n"
            "\n"
            "ReportSpec JSON Schema 详情参见 benchmark-report skill 文档。\n"
        )

    @staticmethod
    def _inject_report_data(html: str, spec: dict[str, Any]) -> str:
        """Inject the full ReportSpec (minus case_results) as a JSON script tag.

        The LLM generates an HTML template with a placeholder
        ``<script type="application/json" id="report-data">{}</script>``
        tag. This method replaces the placeholder content with the real
        ReportSpec JSON (excluding ``case_results``, which is handled by
        ``_fix_case_data`` via the ``cases-data`` tag).

        If the LLM did not generate the tag at all, inject one before
        ``</body>`` so the report's JS can find the data.
        """
        # Build report data without case_results (those go in cases-data tag)
        report_data = {k: v for k, v in spec.items() if k != "case_results"}
        fixed_json = json.dumps(report_data, ensure_ascii=False)
        # Escape </ to prevent premature <script> tag close
        fixed_json = fixed_json.replace("</", "<\\/")
        tag = '<script type="application/json" id="report-data">' + fixed_json + "</script>"

        m = _REPORT_DATA_PATTERN.search(html)
        if m:
            return html[: m.start()] + tag + html[m.end() :]

        # LLM didn't include the report-data tag — inject one before
        # </body> (or at end of HTML) so JS that reads
        # document.getElementById('report-data') doesn't crash.
        logger.warning(
            "webdev_client.inject_report_data_tag "
            "— LLM HTML lacked <script id=report-data>, injecting fallback"
        )
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            return html[: body_close.start()] + tag + html[body_close.start() :]
        return html + tag

    @staticmethod
    def _validate_html_completeness(html: str) -> list[str]:
        """Check that the LLM-generated HTML has all required structural tags.

        Returns a list of missing tag names. If the list is non-empty,
        the HTML was truncated (e.g. only CSS, no ``<body>`` or ``</html>``)
        and must not be uploaded to OSS.
        """
        lower = html.lower()
        return [tag for tag in _REQUIRED_HTML_TAGS if tag not in lower]

    @staticmethod
    def _validate_js_scripts(html: str) -> list[str]:
        """Check JS syntax of executable ``<script>`` blocks with the Node.js parser.

        LLM render scripts occasionally have a syntax flaw (online report
        search_eval-20260911: an extra ``}`` at the end of the main render script, a
        whole-block SyntaxError at browser parse time, so no render function runs —
        report-data/cases-data were injected and the page was still an empty template).
        A parse-time error cannot be caught at runtime, so it must be intercepted before
        the OSS upload (and regenerated under ``_GENERATION_MAX_ATTEMPTS``).

        Skip ``type="application/json"`` data tags.

        Returns:
            A list of problem descriptions; empty means pass.
        """
        node = shutil.which("node")
        if node is None:
            logger.warning("webdev_client.js_syntax_check_skipped node_not_found")
            return []

        problems: list[str] = []
        for idx, m in enumerate(_SCRIPT_TAG_RE.finditer(html)):
            if "application/json" in m.group(1).lower():
                continue
            content = m.group(2)
            if not content.strip():
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as tmp:
                tmp.write(content)
                tmp.flush()
                proc = subprocess.run(  # noqa: S603
                    [node, "--check", tmp.name],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=10,
                )
            if proc.returncode != 0:
                html_line = html[: m.start(2)].count("\n") + 1
                lines = [line.strip() for line in (proc.stderr or proc.stdout).splitlines()]
                detail = next(
                    (line for line in lines if "SyntaxError" in line),
                    next((line for line in lines if "Error" in line), "node --check failed"),
                )
                problems.append(f"script#{idx} (html_line={html_line}): {detail}")
        return problems

    @staticmethod
    def _parse_json_tag(
        html: str,
        pattern: "re.Pattern[str]",
        tag_name: str,
    ) -> tuple[Any, str | None]:
        """Parse a data-tag's JSON (unescape ``<\\/``); return ``(parsed, error)``."""
        m = pattern.search(html)
        if not m:
            return None, f"missing <script id={tag_name}> tag"
        try:
            return json.loads(m.group(1).replace("<\\/", "</")), None
        except json.JSONDecodeError as exc:
            return None, f"{tag_name} JSON parse failed: {exc}"

    @staticmethod
    def _check_data_tags(html: str, case_results: list[Any]) -> list[str]:
        """Data tags are non-empty and parseable (report-data key fields plus a cases-data length reconciliation)."""
        problems: list[str] = []
        report_data, err = WebdevReportClient._parse_json_tag(
            html, _REPORT_DATA_PATTERN, "report-data"
        )
        if err:
            problems.append(err)
        else:
            for key in ("final_score", "total_samples"):
                if key not in report_data:
                    problems.append(f"report-data missing key: {key}")
        # When case_results is empty, _fix_case_data does not inject the cases-data tag; that absence is legal.
        # When there is data, the tag must exist and its length must reconcile with PG.
        if case_results:
            cases_data, err = WebdevReportClient._parse_json_tag(
                html, _CASES_DATA_PATTERN, "cases-data"
            )
            if err:
                problems.append(err)
            elif not isinstance(cases_data, list):
                problems.append("cases-data is not a JSON array")
            elif len(cases_data) != len(case_results):
                problems.append(
                    f"cases-data length {len(cases_data)} != "
                    f"spec.case_results length {len(case_results)}"
                )
        return problems

    @staticmethod
    def _normalize_title_text(text: str) -> str:
        """Normalize a title for matching: drop all whitespace and lowercase.

        The LLM freely spaces titles (a spaced "root cause × dimension" versus the same
        title without spaces, or "last 7 days trend" with or without the space). A
        character-by-character substring match would treat a faithfully rendered section
        as a dropped column (false regeneration in ieval_external-20260914-q4).
        """
        return re.sub(r"\s+", "", text).lower()

    @staticmethod
    def _title_context_hint(html: str, title: str) -> str:
        """When a title is missing, return a snippet around the anchor words so a truly dropped column can be told from a wording difference."""
        anchor_m = re.search(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", title)
        if not anchor_m:
            return ""
        idx = html.find(anchor_m.group(0))
        if idx < 0:
            return ""
        snippet = re.sub(r"\s+", " ", html[max(0, idx - 20) : idx + 60]).strip()
        return f" (near: …{snippet}…)"

    @staticmethod
    def _has_renderable_section_title(html: str, visible_text: str, title: str) -> bool:
        """The title exists in real text, or inside a script HTML fragment.

        The main report template builds sections from JS strings, so the static DOM is
        not enough. An ordinary JS copy array (such as ``const labels=['effect metrics']``)
        is not a section.

        A title container may contain inline children such as a badge. The LLM often writes
        ``<div class="sec">failed attribution <span class="badge">…</span></div>``
        (ieval_external-20260915: the old regex required the title to close immediately,
        a false hit triggered regeneration, and the second request then hit the webdev
        external-link red line and was rejected wholesale).
        """
        norm = WebdevReportClient._normalize_title_text
        if norm(title) in norm(visible_text):
            return True
        compact_title = r"\s*".join(map(re.escape, title))
        return bool(
            re.search(
                rf"<([a-zA-Z][\w:-]*)\b[^>]*>\s*{compact_title}[\s\S]*?</\1\s*>",
                html,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _check_section_presence(
        html: str,
        spec: dict[str, Any],
        case_results: list[Any],
    ) -> list[str]:
        """Whether a data section exists (real node text, or a renderable HTML template fragment)."""
        inspector = _ReportHtmlParser()
        inspector.feed(html)
        problems: list[str] = []
        for title in _REQUIRED_SECTION_TITLES:
            if not WebdevReportClient._has_renderable_section_title(
                html, inspector.visible_text, title
            ):
                problems.append(f"missing required section: {title}")
        for field, title in _CONDITIONAL_SECTION_TITLES:
            if spec.get(field) and not WebdevReportClient._has_renderable_section_title(
                html, inspector.visible_text, title
            ):
                problems.append(
                    f"missing section with data ({field}): {title}"
                    + WebdevReportClient._title_context_hint(html, title),
                )
        if case_results and not WebdevReportClient._has_renderable_section_title(
            html, inspector.visible_text, "case 明细"
        ):
            problems.append("missing section with data (case_results): CASE 明细")
        return problems

    @staticmethod
    def _check_hidden_sections(html: str) -> list[str]:
        """Hidden anti-pattern: a real node or a simple CSS selector hides a data section."""
        problems: list[str] = []
        section_titles = (
            "效果指标",
            "性能 & 效率",
            "档位分布",
            "评测维度",
            "近 7 天趋势",
            "不合格归因",
            "case 明细",
        )
        inspector = _ReportHtmlParser()
        inspector.feed(html)

        def _contains_section_title(text: str) -> str | None:
            norm_text = WebdevReportClient._normalize_title_text(text)
            for title in section_titles:
                if WebdevReportClient._normalize_title_text(title) in norm_text:
                    return title
            return None

        def _node_label(node: _HtmlNode) -> str:
            if node.attrs.get("id"):
                return f"#{node.attrs['id']}"
            classes = node.attrs.get("class", "").split()
            return "." + classes[0] if classes else node.tag

        hidden_ids: set[str] = set()
        hidden_classes: set[str] = set()
        for style_m in _STYLE_BLOCK_RE.finditer(html):
            for rule_m in _CSS_HIDDEN_RE.finditer(style_m.group(1)):
                for raw_sel in rule_m.group(1).split(","):
                    sel = raw_sel.strip()
                    if re.fullmatch(r"#[A-Za-z0-9_-]+", sel):
                        hidden_ids.add(sel[1:])
                    elif re.fullmatch(r"\.[A-Za-z0-9_-]+", sel):
                        hidden_classes.add(sel[1:])

        for node in inspector.nodes:
            title = _contains_section_title(node.text)
            if not title:
                continue
            style = node.attrs.get("style", "")
            inline_hidden = bool(
                re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", style, re.I)
            )
            attr_hidden = "hidden" in node.attrs or node.attrs.get("aria-hidden") == "true"
            id_hidden = bool(node.attrs.get("id") and node.attrs["id"] in hidden_ids)
            classes = set(node.attrs.get("class", "").split())
            class_hidden = bool(classes & hidden_classes)
            if inline_hidden or attr_hidden or id_hidden or class_hidden:
                problems.append(
                    f"data section hidden: {_node_label(node)} contains section title 「{title}」"
                )
        return problems

    @staticmethod
    def _script_text_by_id(html: str, script_id: str) -> str:
        m = re.search(
            rf'<script\b[^>]*\bid="{re.escape(script_id)}"[^>]*>(.*?)</script>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1) if m else ""

    @staticmethod
    def _style_text_by_id(html: str, style_id: str) -> str:
        m = re.search(
            rf'<style\b[^>]*\bid="{re.escape(style_id)}"[^>]*>(.*?)</style>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1) if m else ""

    @staticmethod
    def _check_report_patch_contracts(html: str, spec: dict[str, Any]) -> list[str]:  # noqa: PLR0912
        """Check the key contracts of caller-injected patches, so data is not present but unfilled at runtime.

        Covers incidents such as ieval_external-20260916: the patch script is missing, the
        Case UX hooks are incomplete, or ``renderTable`` row binding still uses the outer
        loop variable ``i``.
        """
        problems: list[str] = []
        case_results = spec.get("case_results") or []
        has_cases = bool(case_results)
        has_trace_cases = any(
            isinstance(case, dict) and case.get("trace_id") and case.get("trace_url")
            for case in case_results
        )

        if has_cases:
            if 'id="query-ux-patch"' not in html and "id='query-ux-patch'" not in html:
                problems.append("missing query-ux-patch for query/output hover")
            report_ux = WebdevReportClient._script_text_by_id(html, "report-ux-patch-js")
            if not report_ux:
                problems.append("missing report-ux-patch-js for case column reshape")
            else:
                if "removeColByName(table,'用例ID')" not in report_ux:
                    problems.append("report-ux-patch-js lacks 用例ID column removal")
                if "findCaseTable" not in report_ux:
                    problems.append("report-ux-patch-js lacks findCaseTable fallback")
                if (
                    "window.filteredCases" not in report_ux
                    or "filtered[pageOffset+i]" not in report_ux
                ):
                    problems.append("report-ux-patch-js lacks filteredCases row fallback")
            query_ux = WebdevReportClient._script_text_by_id(html, "query-ux-patch-js")
            if query_ux and "基本信息" not in query_ux:
                problems.append("query-ux-patch-js lacks rich dropdown 基本信息")
            if _RENDER_TABLE_ROW_BIND_RE.search(html):
                problems.append(
                    "renderTable row bind still uses outer loop index i "
                    "(rows[idx].addEventListener … })(i))"
                )

        if has_trace_cases:
            trace_patch = WebdevReportClient._script_text_by_id(html, "trace-link-patch")
            if not trace_patch:
                problems.append("missing trace-link-patch for Trace链接 rendering")
            elif (
                "window.filteredCases" not in trace_patch
                or "filtered[pageOffset+r]" not in trace_patch
            ):
                problems.append("trace-link-patch lacks filteredCases row fallback")

        if spec.get("score_loss_waterfall") and spec.get("criterion_deltas"):
            style = WebdevReportClient._style_text_by_id(html, "report-enhance-patch")
            if "#waterfall-delta-section{display:grid" not in style:
                problems.append("waterfall/delta layout patch missing two-column grid")

        if spec.get("attribution_metrics"):
            report_ux = WebdevReportClient._script_text_by_id(html, "report-ux-patch-js")
            if "<th>分类</th><th>分布</th><th>个数</th><th>百分比</th>" not in (report_ux or ""):
                problems.append("attribution table patch missing four-column header")

        return problems

    @staticmethod
    def _validate_data_sections(html: str, spec: dict[str, Any]) -> list[str]:
        """Data-domain completeness and visibility check (the last gate before the OSS upload).

        Three gates (any failure rejects the HTML and regenerates):
        1. Data tags are non-empty and parseable (``_check_data_tags``)
        2. Data sections exist (``_check_section_presence``)
        3. Hidden anti-pattern (``_check_hidden_sections``) — a stylesheet ``display:none``
           hits an element that contains the title text. JS ``style.display=''`` clearing
           the inline style does not override a stylesheet rule (online incident
           ieval_external-20260914: tier distribution / trend / attribution / CASE details
           stayed blank forever, while data injection was fine).

        Returns:
            A list of problem descriptions; empty means pass.
        """
        case_results = spec.get("case_results") or []
        return (
            WebdevReportClient._check_data_tags(html, case_results)
            + WebdevReportClient._check_section_presence(html, spec, case_results)
            + WebdevReportClient._check_hidden_sections(html)
            + WebdevReportClient._check_report_patch_contracts(html, spec)
        )

    @staticmethod
    def _normalize_case_io(
        case_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize input/output/expected to strings for JS rendering.

        PG ``input``/``output`` columns are JSONB and may store dicts or
        lists.  When injected into the ``cases-data`` JSON script tag and
        parsed by JS, these become JS objects.  JS
        ``element.textContent = case.input`` on an object produces
        ``[object Object]``.  This method converts non-string values to
        pretty-printed JSON strings before injection.
        """
        text_fields = ("input", "output", "expected", "display_input", "display_output")
        normalized: list[dict[str, Any]] = []
        for case in case_results:
            nc = dict(case)
            for field in text_fields:
                val = nc.get(field)
                if val is None or isinstance(val, str):
                    continue
                if isinstance(val, (dict, list)):
                    nc[field] = json.dumps(val, ensure_ascii=False, indent=2)
                else:
                    nc[field] = str(val)
            normalized.append(nc)
        return normalized

    @staticmethod
    def _fix_case_data(html: str, spec: dict[str, Any]) -> str:
        """Re-inject the original case_results JSON into the generated HTML.

        The LLM may abbreviate or alter case data (e.g. truncating
        input/output with ``...`` ellipsis) while still producing valid JSON.
        To guarantee data honesty, we **unconditionally** replace the
        cases-data script tag content with the original ``case_results``
        from the spec — PG is the single source of truth.

        If the LLM did not generate a ``cases-data`` script tag at all,
        inject one before ``</body>`` so the report's JS can find the data.
        """
        case_results = spec.get("case_results")
        if not case_results:
            return html

        # Normalize input/output/expected to strings — PG JSONB may
        # store them as dicts/lists, which JS renders as [object Object].
        case_results = WebdevReportClient._normalize_case_io(case_results)

        fixed_json = json.dumps(case_results, ensure_ascii=False)
        # Escape </ to prevent premature <script> tag close
        fixed_json = fixed_json.replace("</", "<\\/")
        tag = '<script type="application/json" id="cases-data">' + fixed_json + "</script>"

        m = _CASES_DATA_PATTERN.search(html)
        if m:
            # Replace existing tag content with original case_results.
            return html[: m.start()] + tag + html[m.end() :]

        # LLM didn't include the cases-data tag — inject one before
        # </body> (or at end of HTML) so JS that reads
        # document.getElementById('cases-data') doesn't crash.
        logger.warning(
            "webdev_client.inject_cases_data_tag "
            "— LLM HTML lacked <script id=cases-data>, injecting fallback"
        )
        body_close = _BODY_CLOSE_RE.search(html)
        if body_close:
            return html[: body_close.start()] + tag + html[body_close.start() :]
        return html + tag

    @staticmethod
    def _fill_link_slots(html: str, rule_link_url: str | None = None) -> str:
        """Fill external-link slots: the scoring-rule slot element becomes a real external link (injected by the caller).

        The webdev-agent generation platform's security policy forbids LLM output from
        embedding external links (a violation rejects the whole response). Links the
        report needs stay in the HTML as slots and are filled by this post-process.
        ``rule_link_url`` comes from the run-config snapshot (the Rubric class
        ``rule_link_url`` written through ResolvedProfile). When the URL is missing or
        the slot was not generated, return the HTML unchanged (the slot stays the
        plain-text scoring-rules label, and the report is not blocked).
        """
        if not rule_link_url:
            return html
        if not _RULE_LINK_SLOT_PATTERN.search(html):
            logger.info("webdev_client.link_slot_missing slot=rule-link — skipped")
            return html
        link_html = (
            f'<a href="{_html_escape(rule_link_url, quote=True)}"'
            ' target="_blank" rel="noopener">评分规则 ↗</a>'
        )
        # Callable replacement: a backslash in rule_link_url must not be treated as an escape or backreference by re.sub.
        # (A business-configured URL containing sequences such as \1 / \g would throw bad escape, or produce
        # wrong HTML and fail the whole response, if a string replacement were used).
        return _RULE_LINK_SLOT_PATTERN.sub(lambda _m: link_html, html, count=1)

    @staticmethod
    def _extract_html(response: dict[str, Any]) -> str:
        """Extract HTML text from an A2A JSON-RPC response.

        Walks ``result.artifacts[*].parts[*]`` looking for the first
        ``text`` part.
        """
        result = response.get("result", {})
        artifacts = result.get("artifacts", [])
        for artifact in artifacts:
            parts = artifact.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    return part.get("text", "")
        return ""

    async def _upload_to_storage(
        self,
        spec: dict[str, Any],
        html: str,
    ) -> str | None:
        """Upload HTML to storage and return a 30-day GET presigned URL.

        The object key follows the convention ``{run_id}/report.html``.
        The ``object_key_prefix`` (e.g. ``agent-eval/benchmark-reports``) is
        prepended by ``libs/storage``'s ``build_object_key``.

        Upload alone does not yield a private-bucket accessible URL; after
        ``upload`` we call ``sign_url`` (30 days) for PG
        ``report_meta_json.report_url``.
        """
        if self._storage is None:
            return None

        run_id = _SAFE.sub("_", str(spec.get("run_id", "") or "").strip()) or "unknown"
        key = f"{run_id}/report.html"
        uploaded = await self._storage.upload(
            key,
            html.encode("utf-8"),
            content_type="text/html; charset=utf-8",
        )
        if not uploaded:
            return None

        # Prefer time-limited signed URL for private buckets (CDN concat may 403).
        signed = await self._storage.sign_url(
            key,
            expires_seconds=_REPORT_URL_EXPIRES_S,
        )
        if signed:
            return signed

        logger.warning(
            "webdev_client.storage_sign_url_failed key=%s; falling back to upload URL",
            key,
        )
        return uploaded


__all__ = ["WebdevReportClient"]
