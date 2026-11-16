"""Small stand-in for the OSS SDK. Objects are files under the configured root."""

from benchmark.libs.storage import OSSStorageClient, OSSStorageConfig, StorageError


class Auth:
    def __init__(self, access_key_id: str = "", access_key_secret: str = ""):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret


class StsAuth(Auth):
    def __init__(self, access_key_id: str = "", access_key_secret: str = "", security_token: str = ""):
        super().__init__(access_key_id, access_key_secret)
        self.security_token = security_token


class Bucket:
    def __init__(self, auth: Auth, endpoint: str, bucket_name: str, **kwargs):
        self.auth = auth
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.session = kwargs.get("session")
        self._client = OSSStorageClient(OSSStorageConfig(endpoint=endpoint, bucket=bucket_name))

    def put_object(self, key: str, data, headers=None) -> str:
        del headers
        if self.session is not None:
            response = self.session.do_request(None, 1)
            if getattr(response, "status", 200) >= 400:
                raise exceptions.SignatureDoesNotMatch("signature mismatch")
        body = data.read() if hasattr(data, "read") else data
        return self._client.put_object(key, body)

    def sign_url(self, method: str, key: str, expires: int = 3600, **_kwargs) -> str:
        return self._client.sign_url(method, key, expires)

    def get_object(self, key: str) -> bytes:
        return self._client.get_object(key)


class OssError(StorageError):
    pass


class exceptions:
    OssError = OssError
    NoSuchKey = OssError
    SignatureDoesNotMatch = OssError
