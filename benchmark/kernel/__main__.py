import os

import uvicorn

from benchmark.kernel.api import create_app


def main() -> None:
    port = int(os.environ.get("EVALBENCH_PORT", "8080"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
