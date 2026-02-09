import re
import json

# Strips junk before/after JSON blocks
FENCE_RE = re.compile(r"```(?:json)?|```", flags=re.IGNORECASE)
BRACE_RE = re.compile(r"\{[\s\S]*\}", flags=re.MULTILINE)


def extract_first_json(txt: str):
    """
    Extract and parse the first JSON object from noisy LLM output.

    This function applies a series of cleanups and attempts progressive
    parsing so that a valid summary isn't discarded due to formatting
    noise.

    Args:
        txt (str): Raw text output from the model.

    Returns:
        tuple[dict | None, str | None]: (parsed object, error reason).
            On success: (dict, None).
            On failure: (None, reason string).
    """
    # 1. Types check
    if not isinstance(txt, str):
        return None, "not_string"

    # 2. Removes markdown code fences and surrounding whitespace
    txt = FENCE_RE.sub("", txt)
    txt = txt.strip()

    # 3. Finds the first {...} block in the text
    m = BRACE_RE.search(txt)
    if not m:
        return None, "no_braces"

    blob = m.group(0)

    # 4. Applies common fixes for frequently seen LLM JSON errors:
    #    - Collapses newlines
    #    - Removes trailing commas (invalid JSON)
    #    - Normalizes quote spacing
    blob = blob.replace("\r", "").replace("\n", " ")
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    blob = re.sub(r"(['\"])\s*,\s*(['\"])", r"\1, \2", blob)

    # 5. Attempts parsing; fall back to progressively stripping
    #    non-JSON content from the edges if the first attempt fails.
    try:
        return json.loads(blob), None
    except json.JSONDecodeError:
        # Tries removing junk from edges
        pre_clean  = re.sub(r"^[^{]*", "", txt)
        post_clean = re.sub(r"[^}]*$", "", pre_clean)
        try:
            return json.loads(post_clean), None
        except Exception as e2:
            return None, f"parse_fail:{e2}"
    except Exception as e:
        return None, f"parse_fail:{e}"
