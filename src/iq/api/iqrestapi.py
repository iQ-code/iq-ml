
import requests
import time

debug = False
_base_url = "https://www.inspiration-q.com/api/"
_url_dict = {}
_auth = {}


def initialize_credentials(api_key):
    set_auth(_build_auth(api_key))
    set_url_dict()


def _build_auth(api_key):
    return {
        "Ocp-Apim-Subscription-Key": api_key
    }


def set_url_dict():
    global _url_dict
    _url_dict = _specialize(_base_url, _known_entry_points)


def set_auth(auth):
    global _auth
    _auth = auth


def post(*args, **kwdargs):
    r_json = _post(_base_url, _url_dict, _auth, *args, **kwdargs)
    if 'exception' in r_json:
        raise Exception(r_json['exception'])
    return r_json


def _get(url, headers, **arguments):
    if debug:
        _print_call("GET", url, headers, arguments)
    r = requests.get(url=url, headers=headers, **arguments)
    if debug:
        print(f"Received request Response {r} with content:\n{r.content}")
    if not r.ok:
        raise Exception(f"Error returned from Inspiration-Q API: {r}")
    return r.json()


def _post(base_url, url_dict, auth, function, waittime=1, **arguments):
    headers = auth

    if function not in url_dict:
        error_msg = f"Unknown API function {function}"
        print(url_dict)
        raise Exception(error_msg)

    url = url_dict[function]
    if debug:
        _print_call("POST", url, headers, arguments)
    r = requests.post(url=url, headers=headers, **arguments)
    if debug:
        print(f"Received request Response {r} with content:\n{r.content}")
    if not r.ok:
        error = f"Error returned from Inspiration-Q API. function: {function}. response: {r}"
        if debug:
            print(error)
        raise Exception(error)

    body = r.json()

    if not body.get("status", False):
        return body

    # response has this format
    # {"computationId":"9970243c-44f7-4150-ab81-29c725feeabf","status":"Pending","computationStoreTime":"2024-03-14T17:06:15.452113+00:00"}
    # we need to do a GET to function/{computationId} to retrieve the actual response once calc finishes

    url = base_url + "/" + function + "/" + body["computationId"]

    if debug:
        print(f"GET url for '{function}' is {url}")

    tottime = 0
    API_timeout = 12 * 3600
    valid_solution_identifiers = ["solution", "named_solution", "zscore"]
    while tottime < API_timeout:
        if debug:
            print(f"Waiting for {waittime}s for '{function}' to complete")

        time.sleep(waittime)
        tottime += waittime

        body = _get(url, headers)

        for solution_identifier in valid_solution_identifiers:
            if solution_identifier in body and body[solution_identifier]:
                if debug:
                    print(f"Found body:\n{body}")
                print(f"solution found")
                return body

        if "status" in body and body["status"] == "Failed":
            return body

        waittime = min(2 * waittime, 5)

    if tottime >= API_timeout:
        raise Exception(f"Exceeded maximum time to complete computation with method '{function}'. Maximum time: {API_timeout / 3600:2.1f}h")

    raise Exception(f"Unable to complete computation with method '{function}'")


def _specialize(base_url, entry_points):
    return {k: base_url + "/" + k for k in entry_points}


def _maybe_trim(string, length=69):
    if debug == "short" and len(string) > length:
        return string[:length] + "..." + string[-1]
    return string


def _print_call(type, url, headers, arguments):
    from datetime import datetime, timezone
    print(
        f"Sending {type} request to {url}\nwith header: {headers}\n"
        f"and arguments: {_maybe_trim(str(arguments))}\n"
        f"at timestamp {datetime.now(timezone.utc)}", flush=True)


_known_entry_points = {
    "v1/iq-ml/linear-regression",
    "v1/iq-ml/logistic-regression",
    "v1/iq-ml/sparse-fdr-regression",
}
