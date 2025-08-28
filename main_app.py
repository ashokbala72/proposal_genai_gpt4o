# ------------------------------------------------------------
# Proposal Co-Pilot — Setup + Tab 0 (fixed order)
# ------------------------------------------------------------
from __future__ import annotations
import os, json, hashlib
from pathlib import Path
import streamlit as st
from textwrap import dedent
from dotenv import load_dotenv
import pandas as pd

# ------------------------------------------------------------
# Helpers (all defined before tabs!)
# ------------------------------------------------------------
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
PERSIST_KEYS = [
    "company_profile","customer_overview","rfp_scope","objectives","stakeholders",
    "timeline","competitors","incumbent","constraints","pain_points",
    "win_themes","success_metrics","extras","evaluation_criteria","out_of_scope"
]

def get_data_file(user_id: str) -> str:
    safe_id = user_id.replace("@", "_").replace(".", "_").replace(" ", "_")
    return f"proposal_inputs_{safe_id}.json"

def save_persisted(user_id: str):
    """Save current session state to JSON file."""
    data_file = get_data_file(user_id)
    out = {k: st.session_state.get(k, "") for k in PERSIST_KEYS}
    Path(data_file).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

def gather_all_inputs() -> str:
    """Collect all Tab 0 inputs into a formatted string for prompts."""
    chunks = []
    for k in PERSIST_KEYS:
        v = st.session_state.get(k, "")
        if v:
            chunks.append(f"[{k}] {v}")
    return "\n\n".join(chunks)

def ctx_hash(*parts) -> str:
    """Stable hash of context parts."""
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            continue
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()

def ask_genai(label: str, prompt: str, max_tokens: int = 800) -> str:
    """Call Azure OpenAI safely."""
    if not client:
        return f"[GenAI disabled — missing Azure env values]\n\nPrompt: {prompt[:400]}"
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful proposal assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GenAI call failed: {e}]"

def get_client_name() -> str:
    """Extract client name heuristically from inputs."""
    txt = st.session_state.get("customer_overview","") or st.session_state.get("rfp_scope","")
    if not txt:
        return "Client"
    words = txt.strip().split()
    for w in words:
        if w.istitle():
            return w
    return "Client"


st.set_page_config(page_title="Proposal Co-Pilot GPT-4o", layout="wide")

# ------------------------------------------------------------
# Azure OpenAI client
# ------------------------------------------------------------
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT    = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
AZURE_OPENAI_API_KEY     = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
AZURE_OPENAI_API_VERSION = (os.getenv("AZURE_OPENAI_API_VERSION") or "").strip()
AZURE_OPENAI_DEPLOYMENT  = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or ""
).strip()

if AzureOpenAI and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_VERSION:
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
    except Exception:
        client = None
else:
    client = None




# ------------------------------------------------------------
# Sidebar: User ID + preload persisted data
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### User Profile")
    user_id = st.text_input("Enter your email or username", key="sidebar_user_id")

if user_id:
    data_file = get_data_file(user_id)
    p = Path(data_file)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.items():
                if v is not None:
                    st.session_state[k] = v    # ✅ overwrite even if key exists
            st.sidebar.success(f"Loaded {data_file}")
        except Exception as e:
            st.sidebar.error(f"Error loading saved data: {e}")
    else:
        st.sidebar.info(f"No saved file found: {data_file}")


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
TABS = st.tabs([
    "Proposal Inputs",
    "Market & Competitor Intelligence",
    "Key Aspects of the RFP",
    "Spend",
    "Key Differentiators",
    "Commercial & Pricing Strategy",
    "Proposal Writing & Storyboarding"
])

with TABS[0]:
    st.subheader("Proposal Inputs (persisted)")

    col1, col2 = st.columns(2)

    with col1:
        st.text_area("Your Company Profile",
                     key="company_profile",
                     value=st.session_state.get("company_profile", ""),
                     height=140)
        st.text_area("Customer Overview",
                     key="customer_overview",
                     value=st.session_state.get("customer_overview", ""),
                     height=140)
        st.text_area("RFP Scope",
                     key="rfp_scope",
                     value=st.session_state.get("rfp_scope", ""),
                     height=140)
        st.text_area("Customer Objectives / Outcomes",
                     key="objectives",
                     value=st.session_state.get("objectives", ""),
                     height=140)
        st.text_area("Key Stakeholders & Decision Makers",
                     key="stakeholders",
                     value=st.session_state.get("stakeholders", ""),
                     height=140)
        st.text_area("Expected Timeline / Milestones",
                     key="timeline",
                     value=st.session_state.get("timeline", ""),
                     height=140)

    with col2:
        st.text_area("Known Competitors (comma-separated)",
                     key="competitors",
                     value=st.session_state.get("competitors", ""),
                     height=100)
        st.text_area("Incumbent(s) & Current State",
                     key="incumbent",
                     value=st.session_state.get("incumbent", ""),
                     height=140)
        st.text_area("Constraints / Dependencies",
                     key="constraints",
                     value=st.session_state.get("constraints", ""),
                     height=140)
        st.text_area("Pain Points / Triggers",
                     key="pain_points",
                     value=st.session_state.get("pain_points", ""),
                     height=140)
        st.text_area("Initial Win Themes",
                     key="win_themes",
                     value=st.session_state.get("win_themes", ""),
                     height=140)
        st.text_area("What does success look like?",
                     key="success_metrics",
                     value=st.session_state.get("success_metrics", ""),
                     height=140)
        st.text_area("Any other context",
                     key="extras",
                     value=st.session_state.get("extras", ""),
                     height=100)
        st.text_area("Evaluation Criteria",
                     key="evaluation_criteria",
                     value=st.session_state.get("evaluation_criteria", ""),
                     height=100)
        st.text_area("Out of Scope",
                     key="out_of_scope",
                     value=st.session_state.get("out_of_scope", ""),
                     height=100)

    # Save button
    if st.button("💾 Save All Inputs"):
        if not user_id:
            st.warning("⚠️ Enter a User ID in the sidebar to enable persistence.")
        else:
            try:
                save_persisted(user_id)
                st.success(f"✅ Saved inputs for {user_id} into `{get_data_file(user_id)}`")
            except Exception as e:
                st.error(f"❌ Failed to save data: {e}")

    if user_id:
        st.caption(f"Tab 0 fields will be stored in `{get_data_file(user_id)}`")
    else:
        st.caption("⚠️ Enter a User ID to enable persistence.")


# ------------------------------------------------------------
# 1) Market & Competitor Intelligence
# ------------------------------------------------------------
with TABS[1]:
    st.subheader("Market & Competitor Intelligence")

    def _split_competitors(txt: str):
        raw = [p.strip() for p in (txt or "").replace("\n", ",").split(",")]
        return [r for r in raw if r]

    base_list = _split_competitors(st.session_state.get("competitors", ""))
    DEFAULT_COMPETITORS = ["Accenture","TCS","Infosys","Capgemini","Cognizant","Wipro","IBM Consulting","Deloitte"]

    # Ensure at least 6 competitors
    effective_list = list(base_list)
    seen = {c.lower() for c in effective_list}
    for c in DEFAULT_COMPETITORS:
        if c.lower() not in seen:
            effective_list.append(c)
            seen.add(c.lower())
        if len(effective_list) >= 6:
            break

    effective_competitors = ", ".join(effective_list)
    full_ctx = gather_all_inputs()
    h = ctx_hash(full_ctx, tuple(effective_list), "market_v3_enforced")

    if st.session_state.get("_hash_market") != h or not st.session_state.get("market_intel"):
        base_prompt = dedent(f"""
    Context (inputs + prior outputs):

    {full_ctx}

    TASK — MARKET & COMPETITOR INTELLIGENCE

    RULES:
    - Do NOT include or analyze any areas listed in the "Out of Scope" input.
    - If a competitor is strong in an out-of-scope area, mark it as irrelevant.

    1) Market Snapshot (3–5 bullets): key trends, buying criteria, risks.

    2) Likely Competitors & Their USPs — Markdown table.
       Must include: {", ".join(effective_list)}

       Columns:
       | Competitor | Primary Plays | 3 Specific USPs | Where They Win | Where They’re Weak | Price Posture | Our Counter-Moves |

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
# 3) Spend
# ------------------------------------------------------------
# ------------------------------------------------------------
# 3) Spend (Market & Customer)
# ------------------------------------------------------------
with TABS[3]:
    st.subheader("Spend (Market & Customer)")

    def _to_float(x):
        try:
            s = str(x).replace(",", "").replace("£","").replace("$","").replace("€","").strip()
            if s.endswith("M"): return float(s[:-1]) * 1_000_000
            if s.endswith("B"): return float(s[:-1]) * 1_000_000_000
            return float(s)
        except Exception: 
            return None

    full_ctx   = gather_all_inputs()
    key_aspects= st.session_state.get("rfp_key_aspects", "")
    h = ctx_hash(full_ctx, key_aspects, "spend_json_v5")  # updated hash key

    def _parse_spend_json(txt: str):
        if not txt: return None
        raw = txt.strip()
        if "```" in raw:
            import re
            m = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL)
            if m: raw = m.group(1).strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"): raw.rfind("}")+1]
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        for k in ["market_total_gbp","customer_spend_gbp"]:
            if k in obj: obj[k] = _to_float(obj[k])
        if "share_pct" in obj:
            try: obj["share_pct"] = float(str(obj["share_pct"]).replace("%","").strip())
            except: obj["share_pct"] = None
        return obj

    # 🔄 Refresh button for controlled regeneration
    if st.button("🔄 Refresh Spend Analysis"):
        st.session_state["_hash_spend"] = None
        st.session_state["spend_analysis"] = ""

    if st.session_state.get("_hash_spend") != h or not st.session_state.get("spend_analysis"):
        prompt = dedent(f"""
You are a market analyst.

RULES:
- Exclude all Out of Scope items from analysis.
- Return JSON ONLY (no markdown).
- JSON must have EXACT keys:
  "market_label", "geo", "period", "method",
  "market_total_gbp", "customer_spend_gbp", "share_pct",
  "assumptions", "low_high_range_gbp", "explain", "confidence"

IMPORTANT:
- "assumptions" must be a LIST of key:value strings.
- Each assumption must have BOTH a key and value.
- Example:
  "assumptions": [
    "currency: GBP",
    "items_included: Application migration, Cloud hosting",
    "items_excluded: Data cleansing, Data validation, Training"
  ]

CONTEXT:
{key_aspects}
{full_ctx}
""").strip()



        resp = ask_genai("Spend (JSON)", prompt, max_tokens=800)
        obj = _parse_spend_json(resp)

        if obj:
            st.session_state["spend_analysis"] = json.dumps(obj, indent=2)
            st.session_state["_hash_spend"] = h
        else:
            st.session_state["spend_analysis"] = ""
            st.session_state["_hash_spend"] = h
            st.error("GenAI did not return valid spend JSON.")
            with st.expander("Raw GenAI Output"):
                st.write(resp or "(empty)")

    raw = st.session_state.get("spend_analysis")
    if raw:
        try: data = json.loads(raw)
        except: data = None
    else: 
        data = None

    def _fmt_money(val):
        return f"£{val:,.0f}" if isinstance(val,(int,float)) and val is not None else "—"
    def _fmt_pct(val):
        return f"{val:.1f}%" if isinstance(val,(int,float)) and val is not None else "—"

    if data:
        mt, cs, share = data.get("market_total_gbp"), data.get("customer_spend_gbp"), data.get("share_pct")

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Market total (GBP)", _fmt_money(mt))
        with c2: st.metric(f"{get_client_name()} spend (GBP)", _fmt_money(cs))
        with c3: st.metric("Customer Share", _fmt_pct(share))

        st.markdown(f"**Scope**: {data.get('market_label','—')}  \n"
                    f"**Geo**: {data.get('geo','—')}  \n"
                    f"**Period**: {data.get('period','—')}  \n"
                    f"**Method**: {data.get('method','—')}")

        if isinstance(data.get("low_high_range_gbp"), list) and len(data["low_high_range_gbp"]) == 2:
            lo, hi = _to_float(data["low_high_range_gbp"][0]), _to_float(data["low_high_range_gbp"][1])
            if lo and hi:
                st.caption(f"Range for customer spend: £{lo:,.0f} – £{hi:,.0f}")

        if data.get("assumptions"):
            st.markdown("**Assumptions used**")
            assumptions = data["assumptions"]
            if isinstance(assumptions, str):
                assumptions = [assumptions]  # wrap single string into list
            for a in assumptions:
                st.markdown(f"- {a}")


        if data.get("explain"):
            st.markdown("**How these numbers were derived**")
            st.write(data["explain"])

        # ℹ️ Note about what the spend values mean
        st.info("**Note:** Market Spend = total IT/project market size (not total company revenues). "
                "Client Spend = the client’s spend on IT/project work relevant to this RFP (not their total earnings).")
    else:
        st.info("_No spend analysis yet._")


# ------------------------------------------------------------
# 5) Key Differentiators — client-specific, competitor-aware (GenAI-only)
# ------------------------------------------------------------
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

# --- Tab 5: Commercial & Pricing Strategy (Skill-Rate Aware) ---
# ------------------------------------------------------------
# 5) Commercial & Pricing Strategy
# ------------------------------------------------------------
# ------------------------------------------------------------
# 5) Commercial & Pricing Strategy
# ------------------------------------------------------------
with TABS[5]:
    st.subheader("Commercial & Pricing Strategy (Skill-Rate Aware)")

    # --- JSON extractor helper ---
    def _extract_json_array(txt: str):
        """Extract clean JSON array from GenAI output."""
        if not txt:
            return None
        raw = txt.strip()

        # Remove code fences if present
        if "```" in raw:
            import re
            m = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()

        # Ensure it starts/ends with [ ]
        if "[" in raw and "]" in raw:
            raw = raw[raw.find("["): raw.rfind("]")+1]

        try:
            return json.loads(raw)
        except Exception:
            return None

    # Step 1: Extract roles + rates from GenAI
    if "rate_table" not in st.session_state:
        full_ctx = gather_all_inputs()  # include ALL proposal inputs

        skills_prompt = dedent(f"""
            You are a solution staffing & pricing expert.

            CONTEXT INPUTS (from proposal):
            {full_ctx}

            TASK:
            - Identify ALL delivery roles directly tied to technologies/platforms 
              mentioned in scope, objectives, timeline, or constraints.
            - Roles must explicitly mention the technology (e.g.,
              "SAP ECC ABAP Developer", "ServiceNow ITSM Consultant",
              "Salesforce CRM Architect", "AWS Cloud Solutions Engineer").
            - ❌ Do NOT output generic roles like "Consultant" or "Engineer".
            - For each role, output JSON with:
              * "Skill"
              * "Onshore £/hr"
              * "Offshore £/hr"
              * "Onshore %"
              * "Offshore %"
              * "Hours" = 0
            - If ranges exist (e.g. offshore £38–42/hr, onsite £75–90/hr),
              choose a realistic mid-value.
            - Default mix: Onshore 30%, Offshore 70% if not specified.
            - OUTPUT JSON ARRAY ONLY (no prose, no markdown, no ``` fences).
        """)

        resp = ask_genai("Skill-Rate List", skills_prompt)
        rates = _extract_json_array(resp) or []

        if not rates:
            st.error("❌ GenAI did not return valid skill-role JSON.")
            with st.expander("Raw GenAI output"):
                st.write(resp or "(empty)")
            # Safe fallback
            rates = [
                {"Skill":"(No roles detected — refine Tab 0 inputs)",
                 "Onshore £/hr":0,"Offshore £/hr":0,
                 "Onshore %":0,"Offshore %":0,"Hours":0}
            ]

        st.session_state["rate_table"] = pd.DataFrame(rates)

    # --- Step 2: Show the role table ---
    st.write("### Rate & Role Table")
    st.dataframe(st.session_state["rate_table"], use_container_width=True)

    # --- Step 3: Effort Estimation (Hours) ---
    full_ctx = gather_all_inputs()  # ensure ALL context is passed

    hours_prompt = dedent(f"""
        Context from all captured proposal inputs:
        {full_ctx}

        Roles & Rates:
        {st.session_state['rate_table'].to_dict(orient='records')}

        TASK:
        - Detect project duration (e.g., "Term: 3 years (+2 option)").
        - Allocate realistic hours per role across phases: 
          Migration, Testing, Go-Live, Support.
        - Scale effort according to project duration and technologies.
        - OUTPUT STRICTLY AS JSON ARRAY matching this schema:
          [
            {{"Skill": "...", "Onshore £/hr": 75, "Offshore £/hr": 40,
              "Onshore %": 30, "Offshore %": 70, "Hours": 1200}},
            ...
          ]
        - No prose, no markdown, no comments.
    """)

    resp_hours = ask_genai("Effort Estimation", hours_prompt)
    hours_data = _extract_json_array(resp_hours) or []

    if hours_data:
        df_hours = pd.DataFrame(hours_data)
    else:
        df_hours = st.session_state["rate_table"].copy()
        st.warning("⚠️ Could not parse GenAI hours. Using placeholder 0s.")

    st.write("### Effort-Adjusted Rate Table")
    st.dataframe(df_hours, use_container_width=True)

    # --- Step 4: Pricing Calculation ---
    df_hours["Onshore Cost"] = (
        df_hours["Hours"] * df_hours["Onshore £/hr"] * (df_hours["Onshore %"]/100)
    )
    df_hours["Offshore Cost"] = (
        df_hours["Hours"] * df_hours["Offshore £/hr"] * (df_hours["Offshore %"]/100)
    )
    df_hours["Total Cost"] = df_hours["Onshore Cost"] + df_hours["Offshore Cost"]

    total_cost = df_hours["Total Cost"].sum()

    st.metric("Total Estimated Cost (GBP)", f"£{total_cost:,.0f}")
    st.write("### Final Pricing Table")
    st.dataframe(df_hours, use_container_width=True)

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




