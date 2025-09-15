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
# Tab: Spend (Market & Customer) - ERU Generalized
# ------------------------------------------------------------
with TABS[3]:
    st.subheader("Spend (Market & Customer)")

    import json

    # --- Sector totals (regulator-sourced, 5-year periods annualised) ---
    SECTOR_TOTALS = {
        "water": 104_000_000_000 / 5,   # Ofwat PR24: £104B over 5 years → £20.8B/year
        "elec_dist": 24_800_000_000 / 5, # Ofgem RIIO-ED2: £24.8B (2023–28) → ~£5B/year
        "elec_trans": 30_000_000_000 / 5, # Ofgem RIIO-T2: ~£30B (2021–26) → ~£6B/year
        "gas_dist": 23_900_000_000 / 5,   # Ofgem RIIO-GD2: £23.9B (2021–26) → ~£4.8B/year
        "energy_retail": 65_000_000_000,  # BEIS/market est. ~£60–70B/year
        "renewables": 18_000_000_000,     # BEIS est. ~£15–20B/year
    }

    # --- Company-specific allowances (annualised where known) ---
    COMPANY_ALLOWANCES = {
        # Water (Ofwat PR24)
        "affinity water": 2_300_000_000 / 5,  # £2.3B → ~£460M/year
        "thames water": 18_000_000_000 / 5,   # ~£18B → ~£3.6B/year
        "united utilities": 13_000_000_000 / 5, # ~£13B → ~£2.6B/year

        # Electricity Distribution (RIIO-ED2)
        "uk power networks": 6_800_000_000 / 5, # ~£6.8B → £1.36B/year
        "western power distribution": 7_800_000_000 / 5, # ~£1.56B/year
        "scottish power energy networks": 3_900_000_000 / 5, # ~£780M/year

        # Gas Distribution (RIIO-GD2)
        "cadent gas": 13_000_000_000 / 5,  # ~£2.6B/year
        "northern gas networks": 3_000_000_000 / 5, # ~£600M/year

        # Transmission (RIIO-T2)
        "national grid": 16_000_000_000 / 5, # ~£3.2B/year
    }

    # --- Detect client ---
    client = get_client_name().lower()

    # Default values
    market_total = None
    customer_spend = None
    sector = None

    if "water" in client:
        sector = "water"
        market_total = SECTOR_TOTALS["water"]
        customer_spend = COMPANY_ALLOWANCES.get(client, None)

    elif "power" in client or "distribution" in client:
        sector = "elec_dist"
        market_total = SECTOR_TOTALS["elec_dist"]
        customer_spend = COMPANY_ALLOWANCES.get(client, None)

    elif "grid" in client or "transmission" in client:
        sector = "elec_trans"
        market_total = SECTOR_TOTALS["elec_trans"]
        customer_spend = COMPANY_ALLOWANCES.get(client, None)

    elif "gas" in client:
        sector = "gas_dist"
        market_total = SECTOR_TOTALS["gas_dist"]
        customer_spend = COMPANY_ALLOWANCES.get(client, None)

    else:
        # Retail / Renewables fallback
        if "sse" in client or "centrica" in client or "octopus" in client:
            sector = "energy_retail"
            market_total = SECTOR_TOTALS["energy_retail"]
        elif "renewable" in client or "bp" in client or "shell" in client:
            sector = "renewables"
            market_total = SECTOR_TOTALS["renewables"]

        # Fallback: customer spend from GPT (bounded)
        prompt = f"""
        Estimate annual spend for {client} as a fraction of the UK {sector} market total (£{market_total:,.0f}).
        Keep realistic (1–5% typical).
        Return JSON: {{"customer_spend_gbp": number}}
        """
        resp = ask_genai("Customer Spend JSON", prompt, max_tokens=300)
        try:
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if m:
                customer_spend = json.loads(m.group(0)).get("customer_spend_gbp")
        except:
            customer_spend = None

    # --- Compute share ---
    if market_total and customer_spend:
        share_pct = round(customer_spend / market_total * 100, 1)
        st.session_state["spend_analysis"] = {
            "market_total_gbp": market_total,
            "customer_spend_gbp": customer_spend,
            "share_pct": share_pct,
            "assumptions": [
                f"Market total from regulator ({sector}).",
                f"{client.title()} allowance where published; else GPT-estimated."
            ],
            "confidence": "High" if client in COMPANY_ALLOWANCES else "Medium"
        }
    else:
        st.error("❌ Could not resolve spend data for this client.")
        st.stop()

    # --- Display results ---
    data = st.session_state["spend_analysis"]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Market total (GBP)", f"£{data['market_total_gbp']:,.0f}")
    with c2: st.metric(f"{get_client_name()} spend", f"£{data['customer_spend_gbp']:,.0f}")
    with c3: st.metric("Share %", f"{data['share_pct']:.1f}%")

    st.write("**Assumptions:**", data["assumptions"])
    st.write("**Confidence:**", data["confidence"])

    # --- Layman-friendly explanation ---
    st.markdown(
        f"""
        ### 💡 In simple terms:
        The regulator has set the total market size for the **{sector.replace('_',' ').title()} sector**
        at about **£{data['market_total_gbp']/1_000_000_000:.1f} billion per year**.  

        For **{get_client_name()}**, the allowed or estimated spend is about
        **£{data['customer_spend_gbp']/1_000_000:.0f} million per year**.  

        This means {get_client_name()} controls around **{data['share_pct']}% of the sector’s total spend**.
        """
    )


# ------------------------------------------------------------
# 4) Key Differentiators (Client-Specific)
# ------------------------------------------------------------
with TABS[4]:
    st.subheader("Key Differentiators (Client-Specific)")

    import os
    from openai import AzureOpenAI

    # --- Ensure GenAI client is initialised in session ---
    if "genai_client" not in st.session_state:
        try:
            st.session_state["genai_client"] = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            st.session_state["genai_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        except Exception as e:
            st.error(f"⚠️ Failed to initialise Azure OpenAI client: {e}")
            st.stop()

    def ask_genai(tag, prompt, max_tokens=800, temperature=0):
        """Robust wrapper for Azure OpenAI chat calls"""
        try:
            resp = st.session_state["genai_client"].chat.completions.create(
                model=st.session_state["genai_deployment"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[GenAI call failed: {e}]"

    # --- Gather inputs for context ---
    full_ctx       = gather_all_inputs()
    market_intel   = st.session_state.get("market_intel", "")
    key_aspects    = st.session_state.get("rfp_key_aspects", "")
    spend_analysis = st.session_state.get("spend_analysis", "")
    competitors    = st.session_state.get("competitors", "")
    client_name    = get_client_name()

    # --- Hash to cache results ---
    h = ctx_hash(full_ctx, market_intel, key_aspects, spend_analysis, competitors, "key_differentiators_v3")

    # --- Only regenerate if inputs changed ---
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

    # --- Display results ---
    differentiators_output = st.session_state.get("differentiators", "")
    if differentiators_output and not differentiators_output.startswith("[GenAI call failed"):
        st.markdown(differentiators_output)
    else:
        st.warning("⚠️ Could not generate differentiators. Please check Azure setup or inputs.")

# ------------------------------------------------------------
# Tab: Commercial & Pricing Strategy (Tightened & Sector-Aware)
# ------------------------------------------------------------
with TABS[5]:
    st.subheader("Commercial & Pricing Strategy")

    import pandas as pd, re, json

    # --- Load rate card ---
    try:
        rate_card = pd.read_csv("rate_card.csv")
    except Exception:
        st.error("⚠️ Rate card CSV missing. Using fallback.")
        rate_card = pd.DataFrame([
            {"Skill":"Project Manager","Onshore £/hr":110,"Offshore £/hr":45},
            {"Skill":"SAP ABAP Developer","Onshore £/hr":95,"Offshore £/hr":38},
            {"Skill":"Azure Cloud Architect","Onshore £/hr":120,"Offshore £/hr":50},
            {"Skill":"Data Engineer","Onshore £/hr":100,"Offshore £/hr":40},
            {"Skill":"QA/Test Lead","Onshore £/hr":90,"Offshore £/hr":35},
            {"Skill":"Business Analyst","Onshore £/hr":105,"Offshore £/hr":42},
            {"Skill":"Salesforce Developer","Onshore £/hr":115,"Offshore £/hr":48},
        ])

    rate_card_roles = rate_card["Skill"].tolist()

    # --- Context inputs ---
    scope_text = st.session_state.get("rfp_scope","")
    objectives = st.session_state.get("objectives","")
    constraints = st.session_state.get("constraints","")

    # --- Excluded irrelevant roles ---
    EXCLUDED = ["Mobile Developer", "iOS", "Android", "Game", "Unity", "Frontend (Mobile)"]

    # --- GPT Role Extraction ---
    role_prompt = f"""
    You are identifying IT roles required for an RFP.

    CONTEXT:
    - Sector: Energy, Resources & Utilities (ERU)
    - RFP Scope: {scope_text}
    - Objectives: {objectives}
    - Constraints: {constraints}

    RULES:
    - Only include roles relevant to large-scale IT transformation, data, cloud, ERP, integration, testing, and utilities sector.
    - Do NOT include mobile app developers, game developers, or unrelated roles unless explicitly mentioned.
    - Return JSON only: {{"roles": ["Skill1", "Skill2", ...]}}
    - Use industry-standard IT services role names (SAP ABAP Developer, Azure Architect, Data Engineer, QA/Test Lead, etc.).
    """

    resp_roles = ask_genai("Role Identification", role_prompt, max_tokens=400)

    try:
        roles_json = json.loads(re.search(r"\{.*\}", resp_roles, re.DOTALL).group(0))
        roles_from_gpt = roles_json.get("roles", [])
    except:
        st.warning("⚠️ Could not parse GPT roles, using fallback generic roles.")
        roles_from_gpt = ["Project Manager","Business Analyst","QA/Test Lead"]

    # --- Filter out irrelevant roles ---
    roles_from_gpt = [r for r in roles_from_gpt if not any(x.lower() in r.lower() for x in EXCLUDED)]

    # --- Role Mapping to Rate Card ---
    def map_role_to_ratecard(role, rate_card_roles):
        prompt = f"""
        The role "{role}" is not in our rate card.
        Available rate card roles: {rate_card_roles}

        TASK:
        - Pick the closest matching role from the list.
        - Return JSON only:
        {{"original_role": "{role}", "mapped_role": "ClosestRole"}}
        """
        resp = ask_genai("Role Mapping", prompt, max_tokens=200)
        try:
            obj = json.loads(re.search(r"\{.*\}", resp, re.DOTALL).group(0))
            return obj.get("mapped_role")
        except:
            return None

    roles_final = []
    for role in roles_from_gpt:
        if role in rate_card_roles:
            roles_final.append(role)
        else:
            mapped = map_role_to_ratecard(role, rate_card_roles)
            if mapped:
                roles_final.append(mapped)
            else:
                st.warning(f"⚠️ No mapping found for '{role}'. Please update rate card.")

    roles_final = list(set(roles_final))  # deduplicate
    df_rates = rate_card[rate_card["Skill"].isin(roles_final)].copy()

    # --- Hours Calculation ---
    timeline = st.session_state.get("timeline","12 months")
    months = 12
    m = re.search(r"(\d+)", timeline)
    if m: months = int(m.group(1))
    baseline_hours = months * 160

    prompt_hours = f"""
    Distribute {baseline_hours} hours per role across these roles:
    {roles_final}

    RULES:
    - Return JSON array: [{{"Skill": "...", "Hours": ...}}, ...]
    - Do not invent roles.
    - Hours must sum close to {baseline_hours * len(roles_final)} (±10%).
    """
    resp_hours = ask_genai("Hours Distribution", prompt_hours, max_tokens=400)

    def extract_json_list(text):
        try:
            raw = re.search(r"\[.*\]", text, re.DOTALL).group(0)
            return json.loads(raw)
        except:
            return None

    hours_data = extract_json_list(resp_hours) or []
    if hours_data:
        hours_map = {r["Skill"]: r["Hours"] for r in hours_data if "Skill" in r}
        df_rates["Hours"] = df_rates["Skill"].map(hours_map).fillna(baseline_hours)
    else:
        st.warning("⚠️ GPT did not return hours. Using equal baseline per role.")
        df_rates["Hours"] = baseline_hours

    # --- Costing ---
    df_rates["Onshore %"] = 30
    df_rates["Offshore %"] = 70
    df_rates["Onshore Cost"] = df_rates["Hours"] * df_rates["Onshore £/hr"] * (df_rates["Onshore %"]/100)
    df_rates["Offshore Cost"] = df_rates["Hours"] * df_rates["Offshore £/hr"] * (df_rates["Offshore %"]/100)
    df_rates["Total Cost"] = df_rates["Onshore Cost"] + df_rates["Offshore Cost"]

    # --- Show results ---
    st.write("### Pricing Table with Roles & Hours")
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




