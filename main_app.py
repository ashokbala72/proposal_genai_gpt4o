# ------------------------------------------------------------
# Proposal Co-Pilot: Azure OpenAI Edition (with Save/Retrieve)
# ------------------------------------------------------------

import os
import re
import json
import hashlib
from pathlib import Path
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# ------------------------------------------------------------
# Environment & Azure OpenAI Client
# ------------------------------------------------------------
load_dotenv(override=True)


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-AshokB")

if AzureOpenAI and AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
else:
    client = None

# ------------------------------------------------------------
# Persistence (Tab 0 fields)
# ------------------------------------------------------------
PERSIST_KEYS = [
    "company_profile", "customer_overview", "rfp_scope", "objectives",
    "stakeholders", "timeline", "competitors", "incumbent",
    "constraints", "pain_points", "win_themes", "success_metrics",
    "extras", "client_name", "competitors_override"
]

def get_data_file() -> Path:
    """Return path to JSON file for current email ID."""
    email = st.session_state.get("email_id", "").strip().lower()
    if email:
        safe = re.sub(r"[^a-z0-9]+", "_", email)
        return Path(f"proposal_inputs_{safe}.json")
    else:
        return Path("proposal_inputs.json")

def save_persisted():
    """Save ALL Tab 0 fields into JSON (overwrite)."""
    try:
        out = {k: st.session_state.get(k, "") for k in PERSIST_KEYS}
        get_data_file().write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success(f"Inputs saved for {st.session_state.get('email_id') or 'default user'}")
    except Exception as e:
        st.error(f"Could not save data: {e}")

def load_persisted():
    """Load JSON into session_state (overwrite current values)."""
    try:
        p = get_data_file()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    st.session_state[k] = v
            st.success(f"Loaded saved inputs for {st.session_state.get('email_id') or 'default user'}")
        else:
            st.warning("No saved file found for this email.")
    except Exception as e:
        st.error(f"Could not load data: {e}")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
SYSTEM_PROMPT = """
You are a senior bid manager and market strategist. Write crisp, well-structured, practical outputs.
""".strip()

def ask_gpt(title: str, content: str, temperature: float = None, max_tokens: int = 1400) -> str:
    """Safely call Azure OpenAI and return text (Azure-compliant)."""
    prompt = dedent(content).strip()
    if not client:
        return f"[Offline stub: {title}]\n\n{prompt[:800]}"
    try:
        with st.spinner(f"Generating {title} with GenAI…"):
            kwargs = {
                "model": AZURE_OPENAI_DEPLOYMENT_NAME,
                "max_completion_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            # Only include temperature if provided and supported
            if temperature is not None and AZURE_OPENAI_DEPLOYMENT_NAME not in ["gpt-4o", "gpt-4o-mini", "gpt-4o-raj", "gpt-5-AshokB"]:
                kwargs["temperature"] = temperature

            resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GenAI call failed: {e}]"



def gather_all_inputs() -> str:
    """Concatenate all persisted fields into a single string."""
    data = []
    for k, v in st.session_state.items():
        if isinstance(v, str) and v.strip():
            data.append(f"[{k}]\n{v.strip()}")
    return "\n\n".join(data)

def extract_client_name_from_overview(overview: str) -> str | None:
    if not isinstance(overview, str) or not overview.strip():
        return None
    line1 = overview.strip().splitlines()[0].strip()
    for sep in [":", " - ", "—", "(", ","]:
        if sep in line1:
            line1 = line1.split(sep, 1)[0].strip()
            break
    line1 = re.sub(r"^[\"'«“]+|[\"'»”]+$", "", line1).strip()
    line1 = re.sub(r"\s{2,}", " ", line1).strip()
    return line1 if re.search(r"[A-Za-z]", line1) else None

def get_client_name() -> str:
    if st.session_state.get("client_name"):
        return st.session_state["client_name"]
    ov = st.session_state.get("customer_overview", "")
    derived = extract_client_name_from_overview(ov)
    if derived:
        st.session_state["client_name"] = derived
        return derived
    return "Client"

def ctx_hash(*parts) -> str:
    m = hashlib.md5()
    for p in parts:
        if p is None:
            continue
        m.update(str(p).encode("utf-8", "ignore"))
    return m.hexdigest()

# ------------------------------------------------------------
# App UI
# ------------------------------------------------------------
st.set_page_config(page_title="Proposal Co-Pilot", layout="wide")
st.title("📄 Proposal Co-Pilot — GenAI-assisted RFP Builder")

with st.sidebar:
    st.text_input("User Email ID", key="email_id")
    if st.button("📂 Retrieve Saved Data"):
        load_persisted()
    st.caption("Each email has its own JSON file.")
    st.markdown("---")
    st.markdown("**Azure OpenAI Environment**")
    st.write("Endpoint:", AZURE_OPENAI_ENDPOINT or "(not set)")
    st.write("Deployment:", AZURE_OPENAI_DEPLOYMENT_NAME)
    st.write("API Version:", AZURE_OPENAI_API_VERSION)
    st.write("API Key configured:", bool(AZURE_OPENAI_API_KEY))

TABS = st.tabs([
    "Proposal Inputs",
    "Market & Competitor Intelligence",
    "Key Aspects of the RFP",
    "Spend",
    "Key Differentiators",
    "Commercial & Pricing Strategy",
    "Proposal Writing & Storyboarding",
])

# ------------------------------------------------------------
# 1) Proposal Inputs (persisted per email)
# ------------------------------------------------------------
with TABS[0]:
    st.subheader("Proposal Inputs")

    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Your Company Profile", key="company_profile", height=140)
        st.text_area("Customer Overview", key="customer_overview", height=140)
        st.text_area("RFP Scope", key="rfp_scope", height=140)
        st.text_area("Customer Objectives / Outcomes", key="objectives", height=140)
        st.text_area("Key Stakeholders", key="stakeholders", height=140)
        st.text_area("Timeline / Milestones", key="timeline", height=140)

    with col2:
        st.text_area("Known Competitors", key="competitors", height=100)
        st.text_area("Incumbent(s)", key="incumbent", height=140)
        st.text_area("Constraints / Dependencies", key="constraints", height=140)
        st.text_area("Pain Points", key="pain_points", height=140)
        st.text_area("Win Themes", key="win_themes", height=140)
        st.text_area("Success Metrics", key="success_metrics", height=140)
        st.text_area("Any other context", key="extras", height=100)
        st.text_area("Override Competitors", key="competitors_override", height=100)
        st.text_input("Client Name (override)", key="client_name")

    # Manual save button at the bottom
    if st.button("💾 Save All Inputs"):
        save_persisted()

# ------------------------------------------------------------
# 2) Market & Competitor Intelligence (auto) — enforced inclusion of user competitors
# ------------------------------------------------------------
with TABS[1]:
    from textwrap import dedent
    import re as _re

    st.subheader("Market & Competitor Intelligence")

    # Helpers
    def _split_competitors(txt: str):
        raw = [p.strip() for p in (txt or "").replace("\n", ",").split(",")]
        seen = set()
        out = []
        for r in raw:
            if r and r.lower() not in seen:
                out.append(r)
                seen.add(r.lower())
        return out

    def _too_generic(text: str) -> bool:
        patterns = [
            r"\bCompetitor\s+[A-Z]\b",
            r"\bspecific\s+USP\b",
            r"\bparticular\s+strength\b",
            r"\bStrong\s+regional\s+presence\b",
            r"\bNiche\s+specialization\b",
        ]
        return any(_re.search(p, text or "", flags=_re.IGNORECASE) for p in patterns)

    def _names_present(text: str, names: list[str]) -> set[str]:
        found = set()
        t = text or ""
        for n in names:
            if _re.search(rf"(?i)\b{_re.escape(n)}\b", t):
                found.add(n)
        return found

    DEFAULT_COMPETITORS = [
        "Accenture", "TCS", "Infosys", "Capgemini", "Cognizant",
        "Wipro", "IBM Consulting", "Deloitte"
    ]

    # Inputs from Tab 0 (and optional override shown here only)
    base_competitors_text = st.session_state.get("competitors", "")
    with st.expander("Competitors (auto from Tab 0) — override if needed", expanded=False):
        st.caption("Leave blank to use Tab 0. Typing here will not overwrite Tab 0.")
        override_competitors_text = st.session_state.get("competitors_override", "")


    base_list     = _split_competitors(base_competitors_text)   # user-provided in Tab 0
    override_list = _split_competitors(override_competitors_text)
    must_include  = override_list or base_list                   # MUST be present in the table

    # Build effective list with padding to at least 6 names
    effective_list = list(must_include)
    if len(effective_list) < 6:
        seen = {c.lower() for c in effective_list}
        for c in DEFAULT_COMPETITORS:
            if c.lower() not in seen:
                effective_list.append(c)
                seen.add(c.lower())
            if len(effective_list) >= 6:
                break
    effective_competitors = ", ".join(effective_list)

    # Full context + hashing
    full_ctx = gather_all_inputs()
    h = ctx_hash(full_ctx, tuple(effective_list), "market_v3_enforced")

    if st.session_state.get("_hash_market") != h or not st.session_state.get("market_intel"):
        base_prompt = dedent(f"""
            Context (verbatim captured inputs + prior outputs):

            {full_ctx}

            TASK — MARKET & COMPETITOR INTELLIGENCE

            1) Market Snapshot (3–5 bullets): key trends, typical buying criteria, main risks.

            2) Likely Competitors & Their USPs — produce a MARKDOWN TABLE with one row per competitor.
               YOU MUST include rows for ALL of these user-listed competitors (no exceptions):
               {", ".join(must_include) if must_include else "(none provided)"}

               You may add additional plausible firms from this pool:
               {effective_competitors}

               Table columns (exact):
               | Competitor | Primary Plays in This Scope | 3 Specific USPs | Where They Win | Where They’re Weak | Price Posture | Our Counter-Moves |

               Rules:
               - Do NOT use placeholders like "Competitor A" or vague phrases like "specific USP".
               - Provide concrete, plausible USPs (e.g., SAP S/4HANA migration factory, Snowflake accelerators,
                 Salesforce loyalty expertise, EU managed services scale, FinOps frameworks).
               - Keep claims neutral and non-promotional.

            3) Evidence to Request (5–7 bullets): metrics, references, artifacts we should ask for in clarifications.

            Output must be concise, well-structured, and client-facing.
        """).strip()

        draft = ask_gpt("Market Intelligence", base_prompt)

        # Validate & possibly regenerate if generic or missing required names
        missing = []
        if must_include:
            present = _names_present(draft, must_include)
            missing = [n for n in must_include if n not in present]

        if _too_generic(draft) or missing:
            strict_prompt = base_prompt + dedent(f"""

                REGENERATE STRICT:
                - Ensure the markdown table includes ALL of these required competitors as separate rows:
                  {", ".join(must_include) if must_include else "(none provided)"}
                - Absolutely no generic placeholders; use concrete USPs.
                - Keep the same sections (Market Snapshot, Table, Evidence to Request).
            """)
            draft = ask_gpt("Market Intelligence (strict)", strict_prompt, temperature=0.35)

            # One more check; if still missing, nudge explicitly by name
            if must_include:
                present = _names_present(draft, must_include)
                still_missing = [n for n in must_include if n not in present]
                if still_missing:
                    nudge_prompt = base_prompt + dedent(f"""

                        FINAL ATTEMPT:
                        - You omitted: {", ".join(still_missing)}.
                        - Regenerate the entire output and include a row for EACH of those names in the table.
                        - Keep content concise and professional.
                    """)
                    draft = ask_gpt("Market Intelligence (final enforce)", nudge_prompt, temperature=0.25)

        st.session_state["market_intel"] = draft
        st.session_state["_hash_market"] = h

    st.markdown(st.session_state.get("market_intel", "_Generating…_"))


# ------------------------------------------------------------
# 3) Key Aspects of the RFP (auto)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 3) Key Aspects of the RFP — Win-Critical Distillation (GenAI-only)
# ------------------------------------------------------------
with TABS[2]:
    from textwrap import dedent

    st.subheader("Key Aspects of the RFP (Win-Critical Only)")

    full_ctx = gather_all_inputs()
    market_intel = st.session_state.get("market_intel", "")
    competitors  = st.session_state.get("competitors", "")
    client_name  = st.session_state.get("client_name") or st.session_state.get("customer_name") or "the client"

    h = ctx_hash(full_ctx, market_intel, competitors, "key_aspects_win_critical_v2")

    if st.session_state.get("_hash_key_aspects") != h or not st.session_state.get("rfp_key_aspects"):
        prompt = dedent(f"""
            You are a capture strategist. From all provided context, extract ONLY the **win-critical** aspects of the RFP
            (the few things that most influence scoring and award).

            OUTPUT (Markdown only):
            1) **Top 6–8 Win-Critical Aspects** — a table with columns:
               | Aspect | Why it matters for {client_name} | Evidence we must show | Our positioning (one line) | Risk if weak (L/M/H) | Likely evaluator criterion |
            2) **Not Win-Critical / De-scoped** — 4–6 bullets (items present in inputs but not decisive to win)
            3) **Gaps & Clarifications to Request** — 4–8 bullets
            4) **Eval Mapping** — short table aligning aspects to typical criteria & weights (your best estimate)

            STRICTNESS:
            - Be specific to {client_name} and the captured context — do NOT repeat inputs verbatim.
            - No placeholders (e.g., “specific USP”) or generic consultancy phrases.
            - Use the Market/Competitor intel if helpful to judge what *really* matters.
            - Keep concise; client-facing tone.

            CONTEXT:
            --- MARKET INTEL ---
            {market_intel}

            --- RAW INPUTS & PRIOR OUTPUTS ---
            {full_ctx}
        """).strip()

        with st.spinner("Deriving win-critical aspects…"):
            out = ask_gpt("Key Aspects (Win-Critical)", prompt, temperature=0.3)
        st.session_state["rfp_key_aspects"] = out or ""
        st.session_state["_hash_key_aspects"] = h

    st.markdown(st.session_state.get("rfp_key_aspects", "_Generating…_"))


# ------------------------------------------------------------
# 4) Spend (auto)
# ------------------------------------------------------------
with TABS[3]:
    st.subheader("Market & Customer Spend (Scope Area)")
    full_ctx = gather_all_inputs()
    h = ctx_hash(full_ctx, "spend")
    if st.session_state.get("_hash_spend") != h or not st.session_state.get("spend_analysis"):
        prompt = dedent(f"""
            Using the inputs below, provide a reasoned estimate of MARKET SPEND and CUSTOMER SPEND in the scope area.
            Include: ballpark ranges, key cost drivers, comparable deal archetypes, and how spend scales with scope.
            Avoid external links.

            {full_ctx}
        """).strip()
        st.session_state["spend_analysis"] = ask_gpt("Spend", prompt)
        st.session_state["_hash_spend"] = h

# ------------------------------------------------------------
# 4) Spend — Market & Customer (GenAI JSON, validated)
# ------------------------------------------------------------
with TABS[3]:
    from textwrap import dedent
    import json as _json
    import re as _re

    st.subheader("Spend (Market & Customer)")

    def _to_float(x):
        try:
            s = str(x)
            s = s.replace(",", "").replace("£", "").replace("$", "").replace("€", "").strip()
            # Allow suffixes like B/M if the model uses them
            m = _re.match(r"^([0-9]*\.?[0-9]+)\s*([BbMmKk]?)$", s)
            if m:
                n = float(m.group(1))
                suf = m.group(2).lower()
                if suf == "b": n *= 1_000_000_000
                if suf == "m": n *= 1_000_000
                if suf == "k": n *= 1_000
                return float(n)
            return float(s)
        except Exception:
            return None

    full_ctx   = gather_all_inputs()
    key_aspects = st.session_state.get("rfp_key_aspects", "")
    market_intel = st.session_state.get("market_intel", "")
    client_name = get_client_name()


    # Button to force regeneration
    force_re = st.button("🔁 Re-estimate Spend (GenAI)")

    h = ctx_hash(full_ctx, key_aspects, market_intel, "spend_json_v3")
    need = force_re or (st.session_state.get("_hash_spend") != h) or (not st.session_state.get("spend_analysis"))

    def _parse_spend_json(txt: str):
        """Expect a single JSON object with required keys."""
        if not isinstance(txt, str) or not txt.strip():
            return None
        raw = txt.strip()
        if raw.startswith("```"):
            m = _re.search(r"```(?:json|JSON)?\s*(.*?)```", raw, flags=_re.DOTALL)
            if m:
                raw = m.group(1).strip()
        # If the model returns text + JSON, try to extract the last {...}
        braces = list(_re.finditer(r"\{", raw))
        if braces:
            last_open = braces[-1].start()
            candidate = raw[last_open:]
            # find matching closing
            last_close = candidate.rfind("}")
            if last_close != -1:
                raw = candidate[:last_close+1]
        try:
            obj = _json.loads(raw)
        except Exception:
            return None

        # Normalize numeric fields
        for k in ["market_total_gbp", "customer_spend_gbp"]:
            if k in obj:
                obj[k] = _to_float(obj[k])
        if "share_pct" in obj:
            try:
                obj["share_pct"] = float(str(obj["share_pct"]).replace("%", "").strip())
            except Exception:
                obj["share_pct"] = None
        return obj

    if need:
        base_prompt = dedent(f"""
            You are a market analyst. From the context, estimate **market spend** and **{client_name} spend** for the scoped area.

            OUTPUT — return **JSON ONLY** (no markdown). Required keys:
            - "market_label": short label of the market scope you are measuring (e.g., "UK SAP AMS for Retail")
            - "geo": region/country coverage (e.g., "UK & Ireland")
            - "period": time basis (e.g., "annual 2025")
            - "method": short description of estimation method (e.g., top-down triangulation via benchmarks)
            - "market_total_gbp": number in GBP for total market spend (same period)
            - "customer_spend_gbp": number in GBP for this customer (same scope/period)
            - "share_pct": (customer_spend_gbp / market_total_gbp) * 100
            - "assumptions": 3–6 bullet assumptions (strings)
            - "low_high_range_gbp": [low, high] plausible bounds for customer_spend
            - "explain": 3–6 sentence narrative explaining how you got the numbers
            - "confidence": one of ["low","medium","high"]

            HARD CONSTRAINTS:
            - The **period and scope must match** for both market and customer numbers.
            - **customer_spend_gbp must not exceed market_total_gbp**. If you truly believe it does, set a lower market scope and explain — but then **restate** market_total_gbp accordingly.
            - Keep values **plausible** with respect to the inputs (don’t default to zeros).
            - Currency: GBP.

            CONTEXT (use it to infer; if a detail is missing, state the assumption):
            --- KEY ASPECTS (win-critical) ---
            {key_aspects}

            --- MARKET / COMPETITOR INTEL ---
            {market_intel}

            --- RAW INPUTS & PRIOR OUTPUTS ---
            {full_ctx}
        """).strip()

        strict_prompt = base_prompt + "\n\nREGENERATE STRICTLY AS JSON. Ensure the customer spend is a realistic share of the market and periods/scopes match exactly."

        with st.spinner("Estimating Market & Customer spend…"):
            resp = ask_gpt("Spend (JSON)", base_prompt, temperature=0.2)
            obj  = _parse_spend_json(resp)

            # Validate; if incoherent, retry once strictly
            def _valid(o):
                return o and isinstance(o, dict) and _to_float(o.get("market_total_gbp")) and _to_float(o.get("customer_spend_gbp")) and float(o["customer_spend_gbp"]) <= float(o["market_total_gbp"])

            if not _valid(obj):
                resp2 = ask_gpt("Spend (JSON) — strict", strict_prompt, temperature=0.1)
                obj   = _parse_spend_json(resp2)

        if obj and _to_float(obj.get("market_total_gbp")) and _to_float(obj.get("customer_spend_gbp")):
            st.session_state["spend_analysis"] = _json.dumps(obj, indent=2)
            st.session_state["_hash_spend"] = h
        else:
            st.session_state["spend_analysis"] = ""
            st.session_state["_hash_spend"] = h
            st.error("GenAI did not return coherent spend JSON. Click **Re-estimate Spend** after refining inputs.")
            if resp:
                with st.expander("Last raw output"):
                    st.code(resp, language="json")

    # ---- Render pretty ----
    raw = st.session_state.get("spend_analysis")
    if raw:
        try:
            data = _json.loads(raw)
        except Exception:
            data = None
    else:
        data = None

    if data:
        mt = _to_float(data.get("market_total_gbp"))
        cs = _to_float(data.get("customer_spend_gbp"))
        share = data.get("share_pct")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Market total (GBP)", f"£{mt:,.0f}" if mt else "—")
        with c2: st.metric(f"{client_name} spend (GBP)", f"£{cs:,.0f}" if cs else "—")
        with c3: st.metric("Customer Share", f"{share:.1f}%" if isinstance(share, (int,float)) else "—")

        st.markdown(f"**Scope**: {data.get('market_label','—')}  \n**Geo**: {data.get('geo','—')}  \n**Period**: {data.get('period','—')}  \n**Method**: {data.get('method','—')}")
        if isinstance(data.get("low_high_range_gbp"), list) and len(data["low_high_range_gbp"]) == 2:
            lo = _to_float(data["low_high_range_gbp"][0]); hi = _to_float(data["low_high_range_gbp"][1])
            st.caption(f"Range for customer spend: £{lo:,.0f} – £{hi:,.0f}" if (lo and hi) else "")

        if data.get("assumptions"):
            st.markdown("**Assumptions used**")
            for a in data["assumptions"]:
                st.write(f"- {a}")

        if data.get("explain"):
            st.markdown("**How these numbers were derived**")
            st.write(data["explain"])
    else:
        st.info("_No spend analysis yet._")
    st.markdown(st.session_state.get("spend_analysis", "_Generating…_"))

# ------------------------------------------------------------
# 5) Key Differentiators (auto)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 5) Key Differentiators — client-specific, competitor-aware (GenAI-only)
# ------------------------------------------------------------
with TABS[4]:
    from textwrap import dedent
    import re as _re

    st.subheader("Key Differentiators (Client-Specific)")

    full_ctx       = gather_all_inputs()
    market_intel   = st.session_state.get("market_intel", "")
    key_aspects    = st.session_state.get("rfp_key_aspects", "")
    spend_analysis = st.session_state.get("spend_analysis", "")
    competitors    = st.session_state.get("competitors", "")
    client_name    = st.session_state.get("client_name") or st.session_state.get("customer_name") or "the client"

    h = ctx_hash(full_ctx, market_intel, key_aspects, spend_analysis, competitors, "key_differentiators_v2")

    if st.session_state.get("_hash_diff") != h or not st.session_state.get("differentiators"):
        prompt = dedent(f"""
            You are a proposal strategist. Produce **client-specific differentiators** that map to buyer priorities
            of value, innovation, and risk mitigation. Use competitor context where relevant.

            OUTPUT (Markdown only):
            1) **Prioritised Differentiators Table** — 6–8 rows, columns:
               | Buyer Priority (Value/Innovation/Risk) | Differentiator (specific to {client_name}) | Proof/Evidence | Competitor(s) Most Exposed | Client Impact Metric |
            2) **How We Prove It in the Proposal** — checklist bullets (artifacts, references, demos)
            3) **Top 3 Win Themes** — short bullets, each with a one-line client outcome

            STRICTNESS:
            - Be concrete and tailored to {client_name}; avoid generic phrases or placeholders.
            - If a proof is not in inputs, infer a realistic one and mark it as an assumption.
            - Use Market & Competitor intel to position against named competitors where possible.

            CONTEXT:
            --- MARKET & COMPETITOR INTEL ---
            {market_intel}

            --- KEY ASPECTS (win-critical) ---
            {key_aspects}

            --- SPEND (JSON, if any) ---
            {spend_analysis}

            --- RAW INPUTS ---
            {full_ctx}
        """).strip()

        with st.spinner("Synthesizing differentiators…"):
            out = ask_gpt("Key Differentiators", prompt, temperature=0.3)
        st.session_state["differentiators"] = out or ""
        st.session_state["_hash_diff"] = h

    st.markdown(st.session_state.get("differentiators", "_Generating…_"))


# ------------------------------------------------------------
# 6) Commercial & Pricing Strategy — skill-based market rates (auto)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 6) Commercial & Pricing Strategy — skill-based market rates (auto + AI hours)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 6) Commercial & Pricing Strategy — skill-based market rates (auto + AI hours)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 6) Commercial & Pricing Strategy — skill-based market rates (auto + AI hours with fallback)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 6) Commercial & Pricing Strategy — skill-based market rates (AI-derived hours only)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 6) Commercial & Pricing Strategy — skill-based market rates (AI hours only, JSON + strict retries)
# ------------------------------------------------------------
with TABS[5]:
    from textwrap import dedent
    import pandas as pd
    import re as _re
    import io as _io
    import difflib as _difflib
    import json as _json

    st.subheader("Commercial & Pricing Strategy (Skill-Rate Aware)")

    # ---- Market rate catalog (edit as needed) ----
    DEFAULT_RATES = [
        {"Skill": "SAP S/4HANA Consultant (Functional)", "Onshore £/hr": 120.0, "Offshore £/hr": 48.0, "Hours": 0.0, "Onshore %": 30.0, "Offshore %": 70.0},
        {"Skill": "SAP Basis / Technical",               "Onshore £/hr": 115.0, "Offshore £/hr": 50.0, "Hours": 0.0, "Onshore %": 30.0, "Offshore %": 70.0},
        {"Skill": "Snowflake Data Engineer",             "Onshore £/hr": 110.0, "Offshore £/hr": 55.0, "Hours": 0.0, "Onshore %": 25.0, "Offshore %": 75.0},
        {"Skill": "Data Architect",                      "Onshore £/hr": 125.0, "Offshore £/hr": 60.0, "Hours": 0.0, "Onshore %": 50.0, "Offshore %": 50.0},
        {"Skill": "Salesforce Consultant",               "Onshore £/hr": 105.0, "Offshore £/hr": 50.0, "Hours": 0.0, "Onshore %": 30.0, "Offshore %": 70.0},
        {"Skill": "Power BI Developer",                  "Onshore £/hr":  90.0, "Offshore £/hr": 40.0, "Hours": 0.0, "Onshore %": 20.0, "Offshore %": 80.0},
        {"Skill": "Cloud Architect (AWS)",               "Onshore £/hr": 130.0, "Offshore £/hr": 60.0, "Hours": 0.0, "Onshore %": 60.0, "Offshore %": 40.0},
        {"Skill": "DevOps / FinOps Engineer",            "Onshore £/hr": 115.0, "Offshore £/hr": 55.0, "Hours": 0.0, "Onshore %": 40.0, "Offshore %": 60.0},
        {"Skill": "Project Manager",                     "Onshore £/hr": 110.0, "Offshore £/hr": 65.0, "Hours": 0.0, "Onshore %": 80.0, "Offshore %": 20.0},
        {"Skill": "Service Desk L2",                     "Onshore £/hr":  45.0, "Offshore £/hr": 20.0, "Hours": 0.0, "Onshore %": 10.0, "Offshore %": 90.0},
    ]

    if "rate_table" not in st.session_state:
        st.session_state["rate_table"] = pd.DataFrame(DEFAULT_RATES)

    st.markdown("**Skill mix & market rates (edit as needed)**")
    rate_df = st.data_editor(
        st.session_state["rate_table"],
        key="rates_editor",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hours": st.column_config.NumberColumn(min_value=0.0, step=10.0),
            "Onshore %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=5.0),
            "Offshore %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=5.0),
            "Onshore £/hr": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="£%0.2f"),
            "Offshore £/hr": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="£%0.2f"),
        },
    )
    st.session_state["rate_table"] = rate_df

    # ---- Re-estimate control ----
    force_reestimate = st.button("🔁 Re-estimate hours (GenAI)")

    # ---- AI effort (hours) — ONLY from GenAI (JSON + strict retries) ----
    full_ctx   = gather_all_inputs()
    skill_list = rate_df["Skill"].astype(str).tolist()
    hours_sum_before = float(rate_df["Hours"].fillna(0).sum())
    effort_hash = ctx_hash(full_ctx, tuple(skill_list), "effort_model_json_v1")

    # derive broad total-effort bounds from timeline keywords (for the prompt only)
    _scope     = (st.session_state.get("rfp_scope") or "").lower()
    _timeline  = (st.session_state.get("timeline") or "").lower()
    txt = " ".join([_scope, _timeline, (st.session_state.get("objectives") or "").lower()])

    months = 0
    weeks  = 0
    mm = _re.findall(r"(\d+)\s*(?:month|months|mo)\b", txt)
    if mm:
        months = max(int(x) for x in mm)
    ww = _re.findall(r"(\d+)\s*(?:week|weeks|wks?)\b", txt)
    if ww:
        weeks = max(int(x) for x in ww)

    # loose bounds (for guidance only; GenAI still decides actual hours)
    if months > 0:
        lower_total, upper_total = months * 160 * 1.5, months * 160 * 6
    elif weeks > 0:
        lower_total, upper_total = weeks * 40 * 1.5,  weeks * 40 * 6
    else:
        lower_total, upper_total = 800, 8000  # generic bounds

    def _norm(s: str) -> str:
        s = (s or "").lower()
        s = _re.sub(r"[^\w\s/+-]", " ", s)
        s = _re.sub(r"\s+", " ", s).strip()
        return s

    def _best_match(name: str, allowed: list) -> str | None:
        if not name:
            return None
        n = _norm(name)
        allowed_norm = [_norm(a) for a in allowed]
        if n in allowed_norm:
            return allowed[allowed_norm.index(n)]
        cand = _difflib.get_close_matches(n, allowed_norm, n=1, cutoff=0.6)
        if cand:
            return allowed[allowed_norm.index(cand[0])]
        ns = set(n.split())
        best, best_score = None, 0.0
        for a, an in zip(allowed, allowed_norm):
            score = len(ns & set(an.split())) / max(1, len(ns | set(an.split())))
            if score > best_score:
                best, best_score = a, score
        return best if best_score >= 0.55 else None

    def _parse_json_attempt(text: str, allowed_skills: list):
        """Parse JSON array of {Skill, Hours, Onshore %, Offshore %}."""
        if not isinstance(text, str) or not text.strip():
            return None, {}
        raw = text.strip()
        # try direct JSON
        candidate = raw
        # strip code fences if any
        if raw.startswith("```"):
            m = _re.search(r"```(?:json|JSON)?\s*(.*?)```", raw, flags=_re.DOTALL)
            if m:
                candidate = m.group(1).strip()
        # fallback: take substring between first '[' and last ']'
        if "[" not in candidate or "]" not in candidate:
            m2 = _re.search(r"\[.*\]", candidate, flags=_re.DOTALL)
            if m2:
                candidate = m2.group(0)
        try:
            data = _json.loads(candidate)
        except Exception:
            return None, {}
        if not isinstance(data, list):
            return None, {}
        # normalize rows
        rows = []
        mapping = {}
        for obj in data:
            if not isinstance(obj, dict):
                continue
            src = obj.get("Skill") or obj.get("Role") or obj.get("Name")
            dst = _best_match(src, allowed_skills)
            if not dst:
                continue
            mapping[src] = dst
            hrs = obj.get("Hours", 0)
            on  = obj.get("Onshore %", obj.get("Onshore", obj.get("OnshorePct", 0)))
            off = obj.get("Offshore %", obj.get("Offshore", obj.get("OffshorePct", 0)))
            # coerce
            def _f(x):
                try:
                    return float(str(x).replace(",", "").replace("%", "").strip())
                except Exception:
                    return 0.0
            rows.append({"Skill": dst, "Hours": _f(hrs), "Onshore %": _f(on), "Offshore %": _f(off)})
        if not rows:
            return None, mapping
        df = pd.DataFrame(rows)
        # combine duplicates
        df = df.groupby("Skill", as_index=False).agg({"Hours":"sum","Onshore %":"mean","Offshore %":"mean"})
        # reindex to given order
        df = df.set_index("Skill").reindex(skill_list).reset_index()
        for c in ["Hours","Onshore %","Offshore %"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # normalize %
        pct = df["Onshore %"] + df["Offshore %"]
        msk = pct > 0
        df.loc[msk, "Onshore %"]  = 100.0 * df.loc[msk, "Onshore %"]  / pct[msk]
        df.loc[msk, "Offshore %"] = 100.0 * df.loc[msk, "Offshore %"] / pct[msk]
        return df, mapping

    def _estimate_hours_genai(skill_list, full_ctx, lower_total, upper_total, attempts=5):
        """Ask GenAI for JSON hours; retry with stricter constraints if zeros/unusable. No local fallback."""
        base = dedent(f"""
            You are a delivery estimation expert.

            OUTPUT FORMAT (MANDATORY):
            - Return JSON ONLY (no markdown, no code fences, no commentary).
            - Top-level JSON must be an array of objects, one per skill, in the SAME ORDER as provided skills.
            - Each object must include keys: "Skill", "Hours", "Onshore %", "Offshore %".
            - "Onshore %" + "Offshore %" must sum to 100 for every row.
            - "Hours" must be a non-negative integer (round if needed).
            - Do NOT output all zeros. Total Hours across all skills MUST be between {int(lower_total)} and {int(upper_total)}.

            SKILLS (use as-is, keep this exact order):
            {skill_list}

            TIMELINE/MILESTONES: {st.session_state.get("timeline", "(not specified)")}
            SCOPE: {st.session_state.get("rfp_scope", "(not specified)")}
            OBJECTIVES: {st.session_state.get("objectives", "(not specified)")}
            CONSTRAINTS/DEPENDENCIES: {st.session_state.get("constraints", "(not specified)")}
            INCUMBENTS/CURRENT STATE: {st.session_state.get("incumbent", "(not specified)")}

            CONTEXT:
            {full_ctx}
        """).strip()

        addenda = [
            "Ensure at least 5 skills have Hours >= 40 unless the explicit scope duration is under 2 weeks.",
            "Re-check: total Hours must be within the stated bounds and not all zeros. Return JSON only.",
            "If a skill is truly out-of-scope, set Hours=0 for that row, but the overall total must still be within bounds.",
            "Do not rename skills. Keep array order identical to the provided list.",
        ]

        raw_attempts = []
        for i in range(attempts):
            prompt = base + ("\n\nENFORCEMENT: " + " ".join(addenda[: i+1]))
            resp = ask_gpt(f"Effort Model (JSON) — attempt {i+1}", prompt, temperature=0.2)
            raw_attempts.append(resp)
            df, _map = _parse_json_attempt(resp, skill_list)
            if df is None:
                continue
            total = float(df["Hours"].sum())
            if total > 0 and (total >= lower_total * 0.75) and (total <= upper_total * 1.25):
                # Accept a bit outside bounds too (±25%) for practicality
                return df, raw_attempts
        return None, raw_attempts

    need_estimate = force_reestimate or (hours_sum_before == 0.0) or (st.session_state.get("_effort_hash") != effort_hash)
    if need_estimate:
        with st.spinner("Estimating hours by skill (GenAI)…"):
            est_df, raw_outs = _estimate_hours_genai(skill_list, full_ctx, lower_total, upper_total, attempts=5)

        if est_df is not None and float(est_df["Hours"].sum()) > 0.0:
            merged = rate_df.merge(
                est_df[["Skill", "Hours", "Onshore %", "Offshore %"]],
                on="Skill",
                how="left",
                suffixes=("", "_est"),
            )
            for c in ["Hours", "Onshore %", "Offshore %"]:
                ec = f"{c}_est"
                if ec in merged.columns:
                    merged[c] = merged[ec].where(merged[ec].notna(), merged[c])
                    merged.drop(columns=[ec], inplace=True)
            st.session_state["rate_table"] = merged
            rate_df = merged
            st.session_state["_effort_hash"] = effort_hash
            st.success("Estimated hours by skill via GenAI (editable above).")
        else:
            st.error("GenAI did not return usable non-zero hours after multiple strict attempts. Adjust inputs (scope/timeline) and click Re-estimate.")
            with st.expander("Show raw model outputs from attempts"):
                for i, raw in enumerate(raw_outs or [], start=1):
                    st.code(raw or "(empty)", language="json")

    # ---------- Calculation: blended cost per skill ----------
    def _f(x, d=0.0):
        try:
            return float(x)
        except Exception:
            return d

    calc = rate_df.copy()
    for col in ["Hours", "Onshore %", "Offshore %", "Onshore £/hr", "Offshore £/hr"]:
        calc[col] = calc[col].apply(_f)

    pct_sum = calc["Onshore %"] + calc["Offshore %"]
    mask = pct_sum > 0
    calc.loc[mask, "Onshore %"]  = 100.0 * calc.loc[mask, "Onshore %"]  / pct_sum[mask]
    calc.loc[mask, "Offshore %"] = 100.0 * calc.loc[mask, "Offshore %"] / pct_sum[mask]

    calc["Blended £/hr"]    = ((calc["Onshore £/hr"] * calc["Onshore %"]) + (calc["Offshore £/hr"] * calc["Offshore %"])) / 100.0
    calc["Resource Cost £"] = calc["Blended £/hr"] * calc["Hours"]
    base_resource_cost      = float(calc["Resource Cost £"].sum())

    # ---------- Non-resource percentages ----------
    cA, cB, cC, cD = st.columns(4)
    with cA:
        overhead_pct = st.number_input("Overhead % (PMO/QA/Compliance)", min_value=0.0, value=15.0, step=1.0)
    with cB:
        tools_pct = st.number_input("Tools/Platforms %", min_value=0.0, value=5.0, step=1.0)
    with cC:
        travel_pct = st.number_input("Travel & Expenses %", min_value=0.0, value=2.0, step=1.0)
    with cD:
        risk_pct = st.number_input("Contingency/Risk %", min_value=0.0, value=10.0, step=1.0)

    nonres_pct        = (overhead_pct + tools_pct + travel_pct + risk_pct) / 100.0
    non_resource_cost = base_resource_cost * nonres_pct
    total_cost        = base_resource_cost + non_resource_cost

    # ---------- KPIs: cost summary ----------
    s1, s2, s3 = st.columns(3)
    s1.metric("Base Resource Cost", f"£{base_resource_cost:,.0f}")
    s2.metric("Non-Resource Cost", f"£{non_resource_cost:,.0f}")
    s3.metric("Total Estimated Cost", f"£{total_cost:,.0f}")

    # ---------- Benchmark & margin ----------
    manual_benchmark = st.text_input("(Optional) Benchmark Price (e.g., £350,000)")
    price_number = None
    if manual_benchmark:
        m = _re.findall(r"([£$€])\s*([0-9,]+(?:\.[0-9]+)?)", manual_benchmark)
        if m:
            price_number = float(m[0][1].replace(",", ""))

    if price_number:
        margin_amt = price_number - total_cost
        margin_pct = (margin_amt / price_number) * 100 if price_number else 0.0
        st.info(
            f"**Margin vs Benchmark**  \n"
            f"- Benchmark Price: £{price_number:,.0f}  \n"
            f"- Total Cost: £{total_cost:,.0f}  \n"
            f"- Margin: £{margin_amt:,.0f} ({margin_pct:.1f}%)"
        )

    # ---------- Feed full context + rate table to GenAI ----------
    rate_csv = calc[[
        "Skill", "Hours", "Onshore %", "Offshore %",
        "Onshore £/hr", "Offshore £/hr", "Blended £/hr", "Resource Cost £"
    ]].to_csv(index=False)

    h = ctx_hash(rate_csv, overhead_pct, tools_pct, travel_pct, risk_pct, manual_benchmark, full_ctx, "commercial_v_json")

    if st.session_state.get("_hash_commercial") != h or not st.session_state.get("commercial_strategy"):
        pricing_prompt = dedent(f"""
            Using the RFP context and the skill-based market rates table below, write a COMMERCIAL & PRICING STRATEGY.

            - Reference key skills (SAP S/4HANA, Snowflake, Salesforce, Cloud, DevOps/FinOps, PM, Service Desk) and comment on whether the rates & onshore/offshore mixes are realistic for UK/EU enterprise deals.
            - Recommend pyramid/mix levers (senior:mid:junior), onshore/offshore shifts by workstream, and how to protect margin without hurting value.
            - Propose a market benchmark price range for this scope and state your assumptions (deal size/term, transition effort, SLA tiers).
            - Suggest 2–3 commercial models (T&M / Fixed / Outcome / Gainshare) with pros/cons for this scenario.
            - End with a compact table:
              | Item | Value |
              | --- | --- |
              | Benchmark Price (single point) | ... |
              | Total Cost | £{total_cost:,.0f} |
              | Expected Margin (Amount) | ... |
              | Expected Margin (%) | ... |

            Skill mix & market rates (CSV):
            {rate_csv}

            Our computed costs:
            - Base Resource Cost: £{base_resource_cost:,.0f}
            - Non-Resource Cost (overhead+tools+travel+risk): £{non_resource_cost:,.0f}
            - Total Estimated Cost: £{total_cost:,.0f}

            (Optional) User-provided benchmark price: {manual_benchmark or "none"}.

            RFP Context (verbatim inputs & earlier AI outputs):
            {full_ctx}
        """).strip()

        st.session_state["commercial_strategy"] = ask_gpt("Commercial & Pricing", pricing_prompt)
        st.session_state["_hash_commercial"] = h

    st.markdown(st.session_state.get("commercial_strategy", "_Generating…_"))

    with st.expander("Cost breakdown by skill (computed)", expanded=False):
        show_cols = [
            "Skill", "Hours", "Onshore %", "Offshore %",
            "Onshore £/hr", "Offshore £/hr", "Blended £/hr", "Resource Cost £"
        ]
        st.dataframe(calc[show_cols], use_container_width=True)


# ------------------------------------------------------------
# 7) Proposal Writing & Storyboarding (auto)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 7) Proposal Writing & Storyboarding — FULL PROPOSAL (section-by-section, robust)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 7) Proposal Writing & Storyboarding — FULL, CLIENT-SPECIFIC BUILDER (with competitor enforcement)
# ------------------------------------------------------------
with TABS[6]:
    from textwrap import dedent
    import re as _re
    import pandas as pd

    st.subheader("Proposal Writing & Storyboarding (Full, Client-Specific)")

    # ---------- Helpers ----------
    def _split_names(txt: str):
        parts = [p.strip() for p in (txt or "").replace("\n", ",").split(",")]
        out, seen = [], set()
        for p in parts:
            if p and p.lower() not in seen:
                out.append(p)
                seen.add(p.lower())
        return out

    def _too_generic(text: str) -> bool:
        bad = [
            r"\bCompetitor\s+[A-Z]\b",
            r"\bspecific\s+USP\b",
            r"\bparticular\s+strength\b",
            r"\bStrong\s+regional\s+presence\b",
            r"\bNiche\s+specialization\b",
            r"\bplaceholder\b",
            r"\bTBD\b",
            r"\bCompany\s+X\b",
        ]
        return any(_re.search(p, text or "", flags=_re.IGNORECASE) for p in bad)

    def _names_present(text: str, names: list[str]) -> set[str]:
        found, t = set(), (text or "")
        for n in names:
            if _re.search(rf"(?i)\b{_re.escape(n)}\b", t):
                found.add(n)
        return found

    # ---------- Inputs & context ----------
    full_ctx            = gather_all_inputs()
    market_intel        = st.session_state.get("market_intel", "")
    rfp_key_aspects     = st.session_state.get("rfp_key_aspects", "")
    spend_analysis      = st.session_state.get("spend_analysis", "")
    differentiators     = st.session_state.get("differentiators", "")
    commercial_strategy = st.session_state.get("commercial_strategy", "")
    competitors_raw     = st.session_state.get("competitors", "")
    client_name         = st.session_state.get("client_name") or st.session_state.get("customer_name") or "Client"

    # Competitors list (must be enforced in the Competitive Edge section)
    must_competitors = _split_names(competitors_raw)

    # Include rate table snapshot if present (for narrative reference, not recalculation)
    rate_csv = ""
    if "rate_table" in st.session_state:
        try:
            rate_csv = st.session_state["rate_table"][[
                "Skill", "Hours", "Onshore %", "Offshore %", "Onshore £/hr", "Offshore £/hr"
            ]].to_csv(index=False)
        except Exception:
            try:
                rate_csv = st.session_state["rate_table"].to_csv(index=False)
            except Exception:
                rate_csv = ""

    # Controls
    colA, colB = st.columns(2)
    with colA:
        tone = st.selectbox(
            "Tone",
            ["Formal & consultative", "Executive crisp", "Technical detail"],
            index=0,
        )
    with colB:
        length_pref = st.selectbox(
            "Length",
            ["Short (6–8 pages)", "Medium (10–14 pages equivalent)", "Long (15–20 pages equivalent)"],
            index=1,
        )

    words_hint = {
        "Short (6–8 pages)": "≈1400–1800 words",
        "Medium (10–14 pages equivalent)": "≈2200–3000 words",
        "Long (15–20 pages equivalent)": "≈3200–4200 words",
    }.get(length_pref, "≈2200–3000 words")

    # Global context blob for all sections
    context_blob = dedent(f"""
        --- FULL CONTEXT (verbatim inputs & upstream AI outputs) ---
        CLIENT NAME: {client_name}

        MARKET & COMPETITOR INTEL:
        {market_intel}

        KEY ASPECTS OF THE RFP:
        {rfp_key_aspects}

        SPEND ANALYSIS:
        {spend_analysis}

        KEY DIFFERENTIATORS (from prior tab):
        {differentiators}

        COMMERCIAL & PRICING STRATEGY (canonical for commercials section):
        {commercial_strategy}

        USER-LISTED COMPETITORS (must be named if referenced): {", ".join(must_competitors) if must_competitors else "(none provided)"}

        RATE TABLE (CSV snapshot if present; for narrative context only):
        {rate_csv}

        RAW INPUT CONTEXT:
        {full_ctx}
    """).strip()

    # Hash to detect upstream changes (forces rebuild)
    base_hash = ctx_hash(context_blob, tone, length_pref, tuple(must_competitors), "full_proposal_client_specific_v2")

    # Section plan (now includes a dedicated Competitor Edge section)
    SECTIONS = [
        ("cover_letter",            "Cover Letter"),
        ("executive_summary",       "Executive Summary"),
        ("client_understanding",    "Understanding of the Client & Current State"),
        ("solution_overview",       "Solution Overview"),
        ("scope_deliverables",      "Scope & Deliverables"),
        ("approach_methodology",    "Approach & Methodology"),
        ("plan_timeline",           "Project Plan & Timeline"),
        ("team_governance",         "Team & Governance"),
        ("risks_mitigations",       "Risks & Mitigations"),
        ("commercials",             "Commercials & Pricing Strategy"),
        ("competitive_edge",        "Competitive Edge & Counter-Moves"),
        ("differentiators",         "Key Differentiators & Proof Points"),
        ("compliance_traceability", "Compliance & Requirements Traceability"),
        ("kpis_service_levels",     "KPIs & Service Levels"),
        ("assumptions_dependencies","Assumptions & Dependencies"),
        ("next_steps",              "Next Steps & Commercial Validity"),
    ]

    # Per-section guidance (client-specific + enforcement)
    SECTION_GUIDANCE = {
        "cover_letter": f"Address {client_name} directly. ≤ 180 words. Reference client context and desired outcomes.",
        "executive_summary": "150–200 words. Summarize client outcomes, why-us, and the value story. Avoid generic claims.",
        "client_understanding": "Use inputs only. If gaps exist, infer from context; call out assumptions explicitly.",
        "solution_overview": "Describe target architecture, delivery model, environments, tooling, and phased roadmap.",
        "scope_deliverables": "Table + bullets of scope and deliverables aligned to the RFP Key Aspects.",
        "approach_methodology": "Discovery → design → build/migrate → test → transition; mention accelerators and standards.",
        "plan_timeline": "Gantt-like table: Phase | Duration | Start | Finish | Key Outputs. Keep dates relative if actuals missing.",
        "team_governance": "Roles, RACI, governance forums, cadence, QA gates. Tie responsibilities to outcomes.",
        "risks_mitigations": "Top 8–10 risks; for each: risk, mitigation, residual risk, owner.",
        "commercials": "Use ONLY the provided Commercial Strategy & rate snapshot; summarise price, cost, margin, assumptions. No new numbers.",
        "competitive_edge": (
            "Write a markdown table naming each competitor from the user list (e.g., TCS, Accenture). "
            "Columns: Competitor | Specific Strengths in This Scope | Where They Win | Where They’re Weak | Our Counter-Moves for This Client. "
            "State concrete, plausible capabilities (e.g., industry accelerators, platform depth, delivery scale) — no placeholders, no generic phrasing."
        ),
        "differentiators": "Tie to market/customer priorities (value, innovation, risk). Give proof points and metrics if available.",
        "compliance_traceability": "Map to requirements at a high level; list clarifications/gaps if any.",
        "kpis_service_levels": "Table with KPI/SLA, target, measurement, cadence, and remediation path.",
        "assumptions_dependencies": "Cross-stream assumptions/dependencies. Keep crisp and verifiable.",
        "next_steps": "Proposal validity, named contacts, and specific clarification asks.",
    }

    # Core generator with strictness + retries for specificity and competitor enforcement
    def gen_section(section_key: str, section_title: str, enforce_competitors: bool = False) -> str:
        base = dedent(f"""
            You are an expert bid writer. Produce the **{section_title}** for a client-ready proposal for **{client_name}**.

            STYLE:
            - Tone: {tone}; UK English; concise and skimmable.
            - Start the output with '## {section_title}'.
            - Use markdown headings, bullet points, and neat tables where helpful.
            - Target overall proposal length {words_hint}; keep this section proportionate.

            SPECIFICITY RULES (MANDATORY):
            - Make content specific to **{client_name}** and the provided inputs/context; avoid placeholders and generic statements.
            - If a detail is missing, infer using GenAI based on the provided context and typical enterprise patterns; **state assumptions** briefly.
            - Absolutely **forbid** phrases like "Competitor A", "specific USP", "particular strength", "placeholder", "TBD", or "Company X".

            SPECIAL INSTRUCTIONS FOR THIS SECTION:
            {SECTION_GUIDANCE.get(section_key, "Write concise, client-facing content for this section.")}

            COMMERCIALS GUARDRAIL (if applicable):
            - Use **Commercial & Pricing Strategy** provided below as the only source of truth for numbers. Do **not** invent new figures.

            CONTEXT (verbatim inputs & upstream outputs):
            {context_blob}
        """).strip()

        addendum = ""
        if enforce_competitors and must_competitors:
            comp_str = ", ".join(must_competitors)
            addendum = dedent(f"""
                COMPETITOR ENFORCEMENT (MANDATORY):
                - You MUST include **each** of the following competitors by name as rows in the table:
                  {comp_str}
                - Provide concrete, plausible strengths/weaknesses for the scope; and **our counter-moves** tailored to **{client_name}**.
                - No generic phrasing; avoid consultancy clichés.
            """).strip()

        prompt = base + ("\n\n" + addendum if addendum else "")

        # Up to 3 attempts: base, strict, final strict
        attempts = [
            prompt,
            prompt + "\n\nREWRITE STRICT: Remove any generic phrasing. Add missing client/competitor specifics.",
            prompt + "\n\nFINAL ENFORCEMENT: Ensure all required competitor names appear and content is concrete and client-specific.",
        ]

        last = ""
        for i, p in enumerate(attempts, start=1):
            out = ask_gpt(f"Proposal Section — {section_title} (try {i})", p, temperature=0.3)
            last = out or ""
            # Minimal sanity: ensure heading present
            if f"## {section_title}" not in last:
                last = f"## {section_title}\n\n" + last.strip()
            if _too_generic(last):
                continue
            if enforce_competitors and must_competitors:
                present = _names_present(last, must_competitors)
                if not all(n in present for n in must_competitors):
                    continue
            # Passed checks
            return last
        # Return last attempt even if imperfect (so user can iterate)
        return last

    # Build/generate each section (cache + spinner)
    compiled_parts, missing_titles = [], []
    for s_key, s_title in SECTIONS:
        sec_hash = ctx_hash(base_hash, s_key)
        cache_key = f"proposal_sec__{s_key}"
        cache_hash_key = f"proposal_sec_hash__{s_key}"

        need = (st.session_state.get(cache_hash_key) != sec_hash) or (not st.session_state.get(cache_key))
        if need:
            with st.spinner(f"Generating: {s_title}…"):
                enforce = (s_key == "competitive_edge")
                content = gen_section(s_key, s_title, enforce_competitors=enforce) or ""
                st.session_state[cache_key] = content
                st.session_state[cache_hash_key] = sec_hash

        part = st.session_state.get(cache_key, "")
        if not part.strip():
            missing_titles.append(s_title)
        compiled_parts.append(part)

    # Compile final proposal
    proposal_title = f"# Proposal Response — {client_name}"
    proposal_md = proposal_title + "\n\n" + "\n\n".join(compiled_parts)

    st.session_state["storyboard"] = proposal_md
    st.session_state["_hash_story_full"] = base_hash

    # Warnings & output
    if missing_titles:
        st.warning("Some sections returned empty or generic content: " + ", ".join(missing_titles))

    st.markdown(st.session_state.get("storyboard", "_Generating…_"))

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Proposal (.md)",
            data=st.session_state["storyboard"].encode("utf-8"),
            file_name=f"proposal_response_{client_name.replace(' ', '_')}.md",
            mime="text/markdown",
        )
    with col2:
        st.download_button(
            label="Download Proposal (.txt)",
            data=st.session_state["storyboard"].encode("utf-8"),
            file_name=f"proposal_response_{client_name.replace(' ', '_')}.txt",
            mime="text/plain",
        )

    # Manual rebuild control
    if st.button("🔁 Rebuild Proposal"):
        for s_key, _ in SECTIONS:
            st.session_state.pop(f"proposal_sec__{s_key}", None)
            st.session_state.pop(f"proposal_sec_hash__{s_key}", None)
        st.experimental_rerun()


