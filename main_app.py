# ------------------------------------------------------------
# Proposal Co-Pilot — GPT-4o Compatible (Azure)
# ------------------------------------------------------------
from __future__ import annotations
import os, re, json, hashlib
from pathlib import Path
from textwrap import dedent
import streamlit as st
from dotenv import load_dotenv
import pandas as pd


# Azure OpenAI
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT   = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
AZURE_OPENAI_API_KEY    = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
AZURE_OPENAI_API_VERSION= (os.getenv("AZURE_OPENAI_API_VERSION") or "").strip()
AZURE_OPENAI_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or ""
).strip()

# Create client
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
# Persistence (Tab 0 fields)
# ------------------------------------------------------------
PERSIST_KEYS = [
    "company_profile","customer_overview","rfp_scope","objectives","stakeholders",
    "timeline","competitors","incumbent","constraints","pain_points",
    "win_themes","success_metrics","extras"
]
DATA_FILE = os.getenv("PROPOSAL_STORE", "proposal_inputs.json")

def load_persisted():
    """Load persisted Tab 0 fields into st.session_state."""
    try:
        p = Path(DATA_FILE)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k in PERSIST_KEYS:
                    if k in data and isinstance(data[k], str):
                        st.session_state.setdefault(k, data[k])
    except Exception as e:
        st.sidebar.warning(f"Could not load persisted data: {e}")

def save_persisted():
    """Save Tab 0 fields back to disk."""
    try:
        out = {k: st.session_state.get(k, "") for k in PERSIST_KEYS}
        Path(DATA_FILE).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        st.sidebar.warning(f"Could not save data: {e}")

def mark_dirty():
    st.session_state["_dirty"] = True

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ctx_hash(*parts) -> str:
    m = hashlib.md5()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            m.update(p)
        else:
            m.update(str(p).encode("utf-8", "ignore"))
    return m.hexdigest()

def gather_all_inputs() -> str:
    """Collect all captured inputs for GenAI."""
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
    line1 = re.sub(r'^["\'«“]+|["\'»”]+$', "", line1).strip()
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
    if st.session_state.get("customer_name"):
        return st.session_state["customer_name"]
    return "Client"

# ------------------------------------------------------------
# GPT-4o Safe ask_genai()
# ------------------------------------------------------------
def ask_genai(title: str, content: str, max_tokens: int | None = 1200) -> str:
    """
    GPT-4o Compatible Azure helper.
      - Uses Chat Completions first
      - Falls back to Responses API if API version >= 2025-03-01-preview
      - Avoids unsupported params
    """
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
    deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, API_VERSION, deployment]):
        return f"[GenAI disabled — missing Azure env values]\n\nPrompt:\n{content[:800]}"

    prompt = (content or "").strip()
    with st.spinner(f"Generating {title} with GPT-4o…"):
        try:
            # --- Try Chat API ---
            kwargs = {"model": deployment, "messages": [{"role":"user","content":prompt}]}
            if max_tokens is not None: kwargs["max_tokens"] = int(max_tokens)
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e1:
            msg = str(e1)
            token_issue = ("Unsupported parameter" in msg and "max_tokens" in msg) or ("max_completion_tokens" in msg)
            if token_issue and API_VERSION >= "2025-03-01-preview":
                try:
                    r = client.responses.create(
                        model=deployment,
                        input=prompt,
                        max_completion_tokens=(max_tokens or 512),
                    )
                    return getattr(r, "output_text", "").strip() or "[GenAI call returned empty content]"
                except Exception as e2:
                    return f"[GenAI error: {e2}]"
            return f"[GenAI error: {e1}]"

# ------------------------------------------------------------
# App Shell
# ------------------------------------------------------------
st.set_page_config(page_title="Proposal Co-Pilot GPT-4o", layout="wide")
st.title("📄 Proposal Co-Pilot — GPT-4o (Azure)")

# Load persisted inputs before UI
load_persisted()

with st.sidebar:
    st.markdown("**Azure Environment**")
    st.code(
        f"Endpoint: {AZURE_OPENAI_ENDPOINT}\n"
        f"API Version: {AZURE_OPENAI_API_VERSION}\n"
        f"Deployment: {AZURE_OPENAI_DEPLOYMENT}\n"
        f"API Key set: {bool(AZURE_OPENAI_API_KEY)}",
        language="text"
    )
    if not client:
        st.warning("Azure client not initialized. Check env variables.")
    st.markdown("---")
    st.markdown("**Persistence**")
    st.caption(f"Tab 0 fields auto-save to `{DATA_FILE}`")


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
    "Proposal Writing & Storyboarding",
])

# ------------------------------------------------------------
# 0) Proposal Inputs (Persisted)
# ------------------------------------------------------------
with TABS[0]:
    st.subheader("Proposal Inputs (persisted)")
    col1, col2 = st.columns(2)

    with col1:
        st.text_area("Your Company Profile", key="company_profile", height=140, on_change=mark_dirty)
        st.text_area("Customer Overview", key="customer_overview", height=140, on_change=mark_dirty)
        st.text_area("RFP Scope", key="rfp_scope", height=140, on_change=mark_dirty)
        st.text_area("Customer Objectives / Outcomes", key="objectives", height=140, on_change=mark_dirty)
        st.text_area("Key Stakeholders & Decision Makers", key="stakeholders", height=140, on_change=mark_dirty)
        st.text_area("Expected Timeline / Milestones", key="timeline", height=140, on_change=mark_dirty)

    with col2:
        _ov = st.session_state.get("customer_overview", "")
        _derived_name = extract_client_name_from_overview(_ov)
        if _derived_name and st.session_state.get("client_name") != _derived_name:
            st.session_state["client_name"] = _derived_name

        st.text_area("Known Competitors (comma-separated)", key="competitors", height=100, on_change=mark_dirty)
        st.text_area("Incumbent(s) & Current State", key="incumbent", height=140, on_change=mark_dirty)
        st.text_area("Constraints / Dependencies", key="constraints", height=140, on_change=mark_dirty)
        st.text_area("Pain Points / Triggers", key="pain_points", height=140, on_change=mark_dirty)
        st.text_area("Initial Win Themes", key="win_themes", height=140, on_change=mark_dirty)
        st.text_area("What does success look like?", key="success_metrics", height=140, on_change=mark_dirty)
        st.text_area("Any other context", key="extras", height=100, on_change=mark_dirty)

    if st.session_state.get("_dirty"):
        save_persisted()
        st.session_state["_dirty"] = False
        st.success("Saved all inputs to disk.")

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

            OUTPUT:
            1) **Top Win-Critical Aspects** — table with columns:
               | Aspect | Why it matters for {client_name} | Evidence to Show | Our Positioning | Risk (L/M/H) | Eval Criterion |
            2) **Not Win-Critical / De-scoped** — bullets
            3) **Clarifications to Request** — bullets
            4) **Eval Mapping** — short table.

            CONTEXT:
            --- MARKET INTEL ---
            {market_intel}
            --- INPUTS ---
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

            STRICT OUTPUT RULES:
            - Return JSON **only** (no markdown, no prose, no ``` fences).
            - JSON must have EXACT keys:
              "market_label", "geo", "period", "method",
              "market_total_gbp", "customer_spend_gbp", "share_pct",
              "assumptions", "low_high_range_gbp", "explain", "confidence"
            - "market_total_gbp" = IT/project market size (not company revenue).
            - "customer_spend_gbp" = client’s spend on IT/project work (not total earnings).
            - "share_pct": number only (no % sign).
            - "low_high_range_gbp": array of 2 numbers.
            - "assumptions": list of strings.
            - "explain": short string.
            - "confidence": one of "low","medium","high".

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
            for a in data["assumptions"]: st.write(f"- {a}")

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

            OUTPUT:
            1) Table with columns:
               | Buyer Priority (Value/Innovation/Risk) | Differentiator (specific to {client_name}) | Proof/Evidence | Competitor Exposed | Client Impact |
            2) How We Prove It — checklist bullets
            3) Top 3 Win Themes — short bullets

            CONTEXT:
            --- MARKET & COMPETITOR INTEL ---
            {market_intel}

            --- WIN-CRITICAL ASPECTS ---
            {key_aspects}

            --- SPEND (JSON) ---
            {spend_analysis}

            --- RAW INPUTS ---
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
            - No placeholders; tailor to context.

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
            10) Next Steps

            If 'Outline only', return bullets under each section.

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

            --- Inputs ---
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




