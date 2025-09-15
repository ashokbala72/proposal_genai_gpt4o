# ------------------------------------------------------------
# Proposal Co-Pilot: Azure OpenAI Edition (Save/Retrieve enabled)
# ------------------------------------------------------------

import os
import re
import json
import hashlib
import pandas as pd
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
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-raj")

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
    "extras", "client_name", "competitors_override",
    "evaluation_criteria", "out_of_scope"
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
            # Only include temperature if supported
            if temperature is not None and AZURE_OPENAI_DEPLOYMENT_NAME not in [
                "gpt-4o", "gpt-4o-mini", "gpt-4o-raj", "gpt-5-AshokB"
            ]:
                kwargs["temperature"] = temperature

            resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GenAI call failed: {e}]"
# Backward compatibility
ask_genai = ask_gpt

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
    """Return either user override, or derived from overview, or fallback."""
    if st.session_state.get("client_name"):
        return st.session_state["client_name"]
    ov = st.session_state.get("customer_overview", "")
    derived = extract_client_name_from_overview(ov)
    if derived:
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
        st.text_area("Evaluation Criteria", key="evaluation_criteria", height=140)
        st.text_area("Out of Scope", key="out_of_scope", height=140)

    with col2:
        st.text_area("Known Competitors", key="competitors", height=100)
        st.text_area("Incumbent(s)", key="incumbent", height=140)
        st.text_area("Constraints / Dependencies", key="constraints", height=140)
        st.text_area("Pain Points", key="pain_points", height=140)
        st.text_area("Win Themes", key="win_themes", height=140)
        st.text_area("Success Metrics", key="success_metrics", height=140)
        st.text_area("Any other context", key="extras", height=100)
        st.text_area("Override Competitors", key="competitors_override", height=100)

        # client_name prefilled but editable
        default_client = extract_client_name_from_overview(
            st.session_state.get("customer_overview", "")
        )
        st.text_input(
            "Client Name (override)",
            key="client_name",
            value=st.session_state.get("client_name", default_client)
        )

    # Manual save button at the bottom
    if st.button("💾 Save All Inputs"):
        save_persisted()
# ------------------------------------------------------------
# Tab: Market & Competitor Intelligence
# ------------------------------------------------------------
with TABS[1]:
    st.subheader("Market & Competitor Intelligence")

    def _split_list(txt: str):
        return [p.strip() for p in (txt or "").replace("\n", ",").split(",") if p.strip()]

    known = _split_list(st.session_state.get("competitors", ""))
    incumbents = _split_list(st.session_state.get("incumbent", ""))
    effective_list = list(dict.fromkeys([c for c in known + incumbents if c]))
    effective_list = [c for c in effective_list if c.lower() != "tcs"]

    full_ctx = gather_all_inputs()
    h = ctx_hash(full_ctx, tuple(effective_list), "market_v4_no_tcs")

    if st.session_state.get("_hash_market") != h or not st.session_state.get("market_intel"):
        base_prompt = dedent(f"""
        Context (inputs + prior outputs):

        {full_ctx}

        TASK — MARKET & COMPETITOR INTELLIGENCE

        RULES:
        - Analyze only competitors provided under Known Competitors or Incumbent(s).
        - Do NOT insert default competitors.
        - Do NOT include TCS.
        - If a competitor is strong in an out-of-scope area, mark it as irrelevant.

        OUTPUT:
        1) Market Snapshot (3–5 bullets).
        2) Likely Competitors & Their USPs — Markdown table.
           Competitors to analyze: {", ".join(effective_list) or "None specified"}
           Columns: | Competitor | Primary Plays | 3 Specific USPs | Where They Win | Where They’re Weak | Price Posture | Our Counter-Moves |
        3) Evidence to Request (5–7 bullets).
        """).strip()

        draft = ask_genai("Market Intelligence", base_prompt)
        st.session_state["market_intel"] = draft
        st.session_state["_hash_market"] = h

    st.markdown(st.session_state.get("market_intel", "_Generating…_"))


# ------------------------------------------------------------
# 2) Key Aspects of the RFP
# ------------------------------------------------------------
with TABS[2]:
    st.subheader("Key Aspects of the RFP (Win-Critical)")

    full_ctx = gather_all_inputs()
    market_intel = st.session_state.get("market_intel", "")
    client_name  = get_client_name()
    h = ctx_hash(full_ctx, market_intel, "key_aspects_win_critical_v2")

    if st.session_state.get("_hash_key_aspects") != h or not st.session_state.get("rfp_key_aspects"):
        prompt = dedent(f"""
    You are a capture strategist. From the context, extract only win-critical aspects.

    RULES:
    - Absolutely ignore anything in "Out of Scope".
    - Out-of-scope items must be listed under "Not Win-Critical / De-scoped".

    OUTPUT:
    1) **Top Win-Critical Aspects** — table with columns:
       | Aspect | Why it matters for {client_name} | Evidence to Show | Our Positioning | Risk (L/M/H) | Eval Criterion |
    2) **Not Win-Critical / De-scoped** — bullets (must include all Out of Scope items)
    3) **Clarifications to Request** — bullets
    4) **Eval Mapping** — short table.

    CONTEXT:
    --- MARKET INTEL ---
    {market_intel}
    --- INPUTS (with Evaluation Criteria & Out of Scope) ---
    {full_ctx}
""").strip()


        out = ask_genai("Key Aspects (Win-Critical)", prompt)
        st.session_state["rfp_key_aspects"] = out
        st.session_state["_hash_key_aspects"] = h

    st.markdown(st.session_state.get("rfp_key_aspects", "_Generating…_"))

# ------------------------------------------------------------
# Tab: Spend (Market & Customer) with Benchmarks + Validation
# ------------------------------------------------------------
with TABS[3]:
    st.subheader("Spend (Market & Customer)")

    import csv, re

    # --- Load UK ERU benchmarks ---
    BENCHMARKS = []
    try:
        with open("uk_eru_benchmarks.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                BENCHMARKS.append(row)
    except Exception:
        st.warning("⚠️ Could not load benchmark CSV. Using GPT-only mode.")

    def _find_benchmark(scope: str):
        for row in BENCHMARKS:
            if row["sector"].lower() in scope.lower():
                return row
        return None

    # --- JSON extractor ---
    def extract_json_block(text: str):
        if not text:
            return None
        m = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        else:
            start, end = text.find("{"), text.rfind("}")
            raw = text[start:end+1] if start != -1 and end != -1 else text
        try:
            return json.loads(raw)
        except Exception:
            return None

    # --- Context ---
    full_ctx = gather_all_inputs()
    key_aspects = st.session_state.get("rfp_key_aspects", "")
    scope_text = st.session_state.get("rfp_scope", "") + " " + st.session_state.get("customer_overview", "")
    benchmark = _find_benchmark(scope_text)
    h = ctx_hash(full_ctx, key_aspects, str(benchmark), "spend_json_v8")

    if st.button("🔄 Refresh Spend Analysis"):
        st.session_state["_hash_spend"] = None
        st.session_state["spend_analysis"] = ""

    if st.session_state.get("_hash_spend") != h or not st.session_state.get("spend_analysis"):
        bench_text = f"Known UK ERU benchmark: {benchmark}" if benchmark else "No benchmark available."
        prompt = dedent(f"""
        You are a market analyst.

        RULES:
        - Exclude all Out of Scope items.
        - Return JSON ONLY. No prose, no markdown, no code fences.
        - Schema must include:
          "market_label", "geo", "period", "method",
          "market_total_gbp", "customer_spend_gbp", "share_pct",
          "assumptions", "low_high_range_gbp", "explain", "confidence"
        - Ensure share_pct = (customer_spend_gbp ÷ market_total_gbp) × 100.
        - Do NOT return values where share_pct > 25% unless explicitly justified.
        - Market totals must align with UK ERU benchmarks if available.

        {bench_text}

        CONTEXT:
        {key_aspects}
        {full_ctx}
        """).strip()

        resp = ask_genai("Spend (JSON)", prompt, max_tokens=800)
        obj = extract_json_block(resp)

        # Retry if failed
        if not obj:
            retry_prompt = prompt + "\n\nYour last output was invalid. Return valid JSON only."
            resp = ask_genai("Spend (Retry JSON)", retry_prompt, max_tokens=800)
            obj = extract_json_block(resp)

        # --- Post-validate & fix share ---
        if obj:
            mt = obj.get("market_total_gbp")
            cs = obj.get("customer_spend_gbp")
            if mt and cs:
                recalculated = round((cs / mt) * 100, 1)
                if abs(recalculated - obj.get("share_pct", 0)) > 1.0:
                    st.warning(f"⚠️ Adjusted share %: recalculated {recalculated}% (was {obj.get('share_pct')}%).")
                    obj["share_pct"] = recalculated

            st.session_state["spend_analysis"] = json.dumps(obj, indent=2)
            st.session_state["_hash_spend"] = h
        else:
            st.error("❌ GenAI did not return valid spend JSON.")
            with st.expander("Raw GenAI Output"):
                st.text(resp or "(empty)")

    raw = st.session_state.get("spend_analysis")
    data = json.loads(raw) if raw else None

    # --- Show Benchmark ---
    if benchmark:
        st.info(f"📊 Benchmark: {benchmark['sector']} ({benchmark['period']}) — "
                f"£{float(benchmark['total_spend_gbp']):,.0f} ({benchmark['notes']})")

    # --- Show Data ---
    if data:
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Market total (GBP)", f"£{data.get('market_total_gbp',0):,.0f}")
        with c2: st.metric(f"{get_client_name()} spend", f"£{data.get('customer_spend_gbp',0):,.0f}")
        with c3: st.metric("Share %", f"{data.get('share_pct',0):.1f}%")
        st.write("**Assumptions:**", data.get("assumptions"))
        st.write("**How derived:**", data.get("explain"))
    else:
        st.info("_No spend analysis yet._")

# ------------------------------------------------------------
# 4) Key Differentiators
# ------------------------------------------------------------
with TABS[4]:
    st.subheader("Key Differentiators (Client-Specific)")

    full_ctx       = gather_all_inputs()
    market_intel   = st.session_state.get("market_intel", "")
    key_aspects    = st.session_state.get("rfp_key_aspects", "")
    spend_analysis = st.session_state.get("spend_analysis", "")
    competitors    = st.session_state.get("competitors", "")
    client_name    = get_client_name()

    h = ctx_hash(full_ctx, market_intel, key_aspects, spend_analysis, competitors, "key_differentiators_v2")

    if st.session_state.get("_hash_diff") != h or not st.session_state.get("differentiators"):
        prompt = dedent(f"""
    You are a proposal strategist. Produce client-specific differentiators.

    RULES:
    - Do NOT generate differentiators for out-of-scope areas.
    - Explicitly tie differentiators and win themes to the Evaluation Criteria input.

    OUTPUT:
    1) Table with columns:
       | Buyer Priority (Value/Innovation/Risk) | Differentiator (specific to {client_name}) | Proof/Evidence | Competitor Exposed | Client Impact | Evaluation Criteria Addressed |
    2) How We Prove It — checklist bullets
    3) Top 3 Win Themes — short bullets, each one MUST clearly state which Evaluation Criterion it addresses.

    CONTEXT:
    --- MARKET & COMPETITOR INTEL ---
    {market_intel}

    --- WIN-CRITICAL ASPECTS ---
    {key_aspects}

    --- SPEND (JSON) ---
    {spend_analysis}

    --- RAW INPUTS (including Evaluation Criteria & Out of Scope) ---
    {full_ctx}
""").strip()




        out = ask_genai("Key Differentiators", prompt)
        st.session_state["differentiators"] = out
        st.session_state["_hash_diff"] = h

    st.markdown(st.session_state.get("differentiators", "_Generating…_"))

# ------------------------------------------------------------
# Tab: Commercial & Pricing Strategy (Controlled Roles + Hours)
# ------------------------------------------------------------
with TABS[5]:
    st.subheader("Commercial & Pricing Strategy (Skill-Rate Aware)")

    import pandas as pd, re

    # Load rate card
    try:
        rate_card = pd.read_csv("rate_card.csv")
    except Exception:
        st.error("⚠️ Rate card CSV missing. Using placeholder rates.")
        rate_card = pd.DataFrame([
            {"Skill":"Project Manager","Onshore £/hr":110,"Offshore £/hr":45},
            {"Skill":"SAP ABAP Developer","Onshore £/hr":95,"Offshore £/hr":38},
            {"Skill":"Azure Cloud Architect","Onshore £/hr":120,"Offshore £/hr":50},
            {"Skill":"Data Engineer","Onshore £/hr":100,"Offshore £/hr":40},
            {"Skill":"QA/Test Lead","Onshore £/hr":90,"Offshore £/hr":35},
            {"Skill":"Business Analyst","Onshore £/hr":105,"Offshore £/hr":42}
        ])

    # Role library (IT-focused)
    ROLE_LIBRARY = {
        "SAP": ["SAP ABAP Developer","SAP FICO Consultant"],
        "Cloud": ["Azure Cloud Architect","DevOps Engineer"],
        "CRM": ["Salesforce Developer","Oracle Utilities Consultant"],
        "Data": ["Data Engineer","BI Consultant","AI/ML Engineer"],
        "Generic": ["Project Manager","Business Analyst","QA/Test Lead"]
    }

    # Always include Generic roles, add others if keywords match
    scope_text = (st.session_state.get("rfp_scope","") + " " + 
                  st.session_state.get("objectives","")).lower()
    roles = set(ROLE_LIBRARY["Generic"])
    for k,v in ROLE_LIBRARY.items():
        if k.lower() in scope_text:
            roles.update(v)

    df_rates = rate_card[rate_card["Skill"].isin(roles)].copy()

    # Hours calculation from timeline (fallback = 12 months)
    timeline = st.session_state.get("timeline","12 months")
    months = 12
    m = re.search(r"(\d+)", timeline)
    if m: months = int(m.group(1))
    baseline_hours = months * 160

    # --- Ask GPT to distribute hours ---
    roles_list = df_rates["Skill"].tolist()
    prompt = f"""
    Distribute {baseline_hours} hours per role across the following roles:
    {roles_list}

    RULES:
    - Return JSON array: [{{"Skill": "...", "Hours": ...}}, ...]
    - Do not invent roles.
    - Hours must sum close to {baseline_hours * len(roles_list)} (±10%).
    """
    resp = ask_genai("Distribute Hours", prompt, max_tokens=400)

    def extract_json_list(text):
        m = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        raw = m.group(1).strip() if m else text
        try: return json.loads(raw)
        except: return None

    hours_data = extract_json_list(resp) or []
    if hours_data:
        hours_map = {r["Skill"]: r["Hours"] for r in hours_data if "Skill" in r}
        df_rates["Hours"] = df_rates["Skill"].map(hours_map).fillna(baseline_hours)
    else:
        st.warning("⚠️ GPT did not return hours. Using equal baseline per role.")
        df_rates["Hours"] = baseline_hours

    # Onshore/Offshore mix (default 30/70, can adjust later)
    df_rates["Onshore %"] = 30
    df_rates["Offshore %"] = 70
    df_rates["Onshore Cost"] = df_rates["Hours"] * df_rates["Onshore £/hr"] * (df_rates["Onshore %"]/100)
    df_rates["Offshore Cost"] = df_rates["Hours"] * df_rates["Offshore £/hr"] * (df_rates["Offshore %"]/100)
    df_rates["Total Cost"] = df_rates["Onshore Cost"] + df_rates["Offshore Cost"]

    # --- Show results ---
    st.write("### Pricing Table with Hours")
    st.dataframe(df_rates, use_container_width=True)
    st.metric("Total Estimated Cost (GBP)", f"£{df_rates['Total Cost'].sum():,.0f}")


# ------------------------------------------------------------
# 6) Proposal Writing & Storyboarding
# ------------------------------------------------------------
with TABS[6]:
    import datetime as _dt
    st.subheader("Proposal Writing & Storyboarding")

    market_intel    = st.session_state.get("market_intel","")
    key_aspects     = st.session_state.get("rfp_key_aspects","")
    differentiators = st.session_state.get("differentiators","")
    spend_analysis  = st.session_state.get("spend_analysis","")
    pricing         = st.session_state.get("commercial_strategy","")
    full_ctx        = gather_all_inputs()
    client_name     = get_client_name()

    c1,c2,c3 = st.columns(3)
    with c1: tone = st.selectbox("Tone", ["Consultative","Crisp/Executive","Technical+Assurance"], index=1)
    with c2: length = st.selectbox("Length", ["Short (2–3p)","Medium (4–6p)","Long (7–10p)"], index=1)
    with c3: include_es = st.checkbox("Include Executive Summary", value=True)

    outline_only = st.checkbox("Generate Outline Only", value=False)
    btn = st.button("✍️ Generate Proposal Draft")

    def _mk_prompt():
        return dedent(f"""
        Create a client-ready proposal draft for **{client_name}**.

        SETTINGS:
        - Tone: {tone}
        - Length: {length}
        - Executive Summary: {"Yes" if include_es else "No"}
        - Format: Markdown with clear headings/bullets.

        REQUIRED SECTIONS:
        1) Executive Summary (if enabled)
        2) Understanding {client_name}'s Priorities
        3) Proposed Solution & Delivery
        4) Differentiators & Proof
        5) Commercials & Pricing Rationale
        6) Risks & Mitigations
        7) Timeline & Milestones
        8) Success Metrics & Value
        9) Assumptions & Dependencies
        10) Out of Scope (explicitly list and state exclusions)
        11) Next Steps

        RULES:
        - Strictly exclude anything listed under "Out of Scope".
        - If an item overlaps, clearly state it in the Out of Scope section.
        - Timelines should always start one week from today.
        - The project must end no later than the date mentioned in "Expected Timeline / Milestones" from Tab 0.
        - Do not use arbitrary earlier dates (e.g., 2024 if current year is 2025).
        - Ensure the Timeline & Milestones section reflects this constraint.
        
        CONTEXT:
        --- Market Intel ---
        {market_intel}

        --- Win-Critical Aspects ---
        {key_aspects}

        --- Differentiators ---
        {differentiators}

        --- Spend ---
        {spend_analysis}

        --- Commercial Strategy ---
        {pricing}

        --- Inputs (including Out of Scope) ---
        {full_ctx}
    """)


    if btn:
        mode = "Storyboard" if outline_only else "Full Draft"
        draft = ask_genai(f"Proposal {mode}", _mk_prompt(), max_tokens=2200)
        st.session_state["proposal_draft"] = draft

    draft_md = st.session_state.get("proposal_draft","")
    if draft_md:
        st.markdown("#### Draft Preview")
        st.markdown(draft_md)
        fname = f"Proposal_{client_name.replace(' ','_')}_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}.md"
        st.download_button("⬇️ Download Markdown", draft_md.encode("utf-8"), file_name=fname, mime="text/markdown")
    else:
        st.info("No draft yet. Fill earlier tabs and click Generate.")




