import streamlit as st
import cv2, yaml, numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

st.set_page_config(page_title="EcoHome Advisor (Safe Mode)", page_icon="🌿", layout="wide")
st.title("🌿 EcoHome Advisor — Safe Mode")

st.markdown("""
**This is a safe-mode build** with extra guards to avoid crashes:
- Defensive YAML parsing
- Defensive image loading & conversion
- No `border=` parameter (for older Streamlit versions)
- Clear on-screen error messages instead of silent failures
""")

DEFAULT_RULES = """
climate_zones:
  hot_humid:
    roof:
      - when: "roof_color == 'dark'"
        recommend: ["High-reflectivity roof coating (SR > 0.8)", "Cool roof membrane / ceramic heat-reflective spray"]
        saving_pct: "8-15%"
        rationale: "High solar radiation + dark roof ⇒ higher heat absorption; cool the roof first."
    walls:
      - when: "humidity == 'high'"
        recommend: ["Mineral wool / wood-fiber insulation (vapor-open)", "Mold-resistant liner / breathable paint"]
        saving_pct: "3-8%"
        rationale: "Hot-humid ⇒ prefer vapor-open systems to reduce moisture/mold risk."
  temperate:
    windows:
      - when: "window_area_ratio > 0.25 and orientation in ['south','west']"
        recommend: ["Low-E window film or double glazing", "External shading (louvers/overhangs/awnings)"]
        saving_pct: "4-10%"
        rationale: "Temperate with strong sun ⇒ reduce solar heat gains first."
  cold_dry:
    windows:
      - when: "window_area_ratio > 0.2 and orientation in ['north','east']"
        recommend: ["Triple glazing (triple-silver Low-E)", "High-airtightness frames", "Improve perimeter sealing"]
        saving_pct: "5-12%"
        rationale: "Cold-dry ⇒ losses through windows dominate; prioritize insulation & airtightness."
budget:
  low: { cap: 1500, prefer: ["Weatherstrips", "Interior shades/curtains", "Localized recycled insulation"] }
  mid:  { cap: 6000, prefer: ["Cool roof coating", "Low-E window film", "External shading kits"] }
  high: { cap: 20000, prefer: ["BIPV roof", "Continuous exterior insulation", "High-performance windows/doors"] }
"""

def safe_parse_yaml(text):
    try:
        return yaml.safe_load(text) or {}
    except Exception as e:
        st.error(f"YAML parse error: {e}")
        return yaml.safe_load(DEFAULT_RULES)

def simple_roof_lightness_score(img_bgr):
    h, w = img_bgr.shape[:2]
    if h < 2 or w < 2:
        return 180.0
    roi = img_bgr[:max(1,h//2), :]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]
    return float(np.mean(L))

def estimate_glass_reflection_ratio(img_bgr):
    try:
        b,g,r = cv2.split(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        diff = cv2.subtract(b, r)
        _, blueish = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(bright, blueish)
        ratio = float(np.sum(mask>0)) / float(img_bgr.shape[0]*img_bgr.shape[1] + 1e-6)
        return ratio
    except Exception:
        return 0.0

def eval_condition(cond:str, ctx:dict):
    try:
        return eval(cond, {}, ctx)
    except Exception:
        return False

def apply_rules(rules:dict, ctx:dict):
    outputs = []
    cz = ctx.get("climate_zone", "temperate")
    zone = (rules.get("climate_zones") or {}).get(cz, {})
    for part, recs in (zone or {}).items():
        if not isinstance(recs, list):
            continue
        for rule in recs:
            cond = str(rule.get("when", "True"))
            if eval_condition(cond, ctx):
                outputs.append({
                    "part": part,
                    "recommend": list(rule.get("recommend") or []),
                    "saving_pct": str(rule.get("saving_pct", "—")),
                    "rationale": str(rule.get("rationale", ""))
                })
    budget_cfg = (rules.get("budget") or {}).get(ctx.get("budget","mid"), {})
    return outputs, budget_cfg

def parse_pct_range(s):
    if not s or s == "—":
        return (0.0, 0.0)
    s = str(s).strip().replace("%","")
    if "-" in s:
        a,b = s.split("-",1)
        try:
            return (float(a), float(b))
        except:
            return (0.0,0.0)
    try:
        v = float(s)
        return (v,v)
    except:
        return (0.0,0.0)

def score_priority(item):
    low, high = parse_pct_range(item.get("saving_pct","0-0"))
    return high

def simple_cost_and_priority(recs, budget_cfg):
    cap = budget_cfg.get("cap", None)
    try:
        cap_val = float(cap) if cap is not None else None
    except:
        cap_val = None
    scored = sorted(recs, key=score_priority, reverse=True)
    items = []
    n = max(len(scored), 1)
    for r in scored:
        low, high = parse_pct_range(r.get("saving_pct","0-0"))
        est_cost = round(cap_val / n) if isinstance(cap_val, float) else None
        items.append({
            "part": r["part"],
            "recommend": r["recommend"],
            "saving": (low, high),
            "est_cost": est_cost,
            "priority": score_priority(r)
        })
    return items

def ai_reasoning(ctx, recs, budget_cfg):
    lines = []
    lines.append(f"- Climate zone: {ctx.get('climate_zone')}  |  Humidity: {ctx.get('humidity')}  |  Orientation: {ctx.get('orientation')}")
    lines.append(f"- Image cues: roof lightness {ctx.get('roof_L'):.1f} ({ctx.get('roof_color')}); glass/reflection {ctx.get('glass_ratio')*100:.1f}%")
    lines.append("- Rule matches:")
    if recs:
        for r in recs:
            lines.append(f"  • [{r['part']}] {', '.join(r['recommend'])} (saving {r['saving_pct']}). Why: {r['rationale']}")
    else:
        lines.append("  • None for current inputs.")
    if budget_cfg:
        lines.append(f"- Budget hint: tier cap ≈ {budget_cfg.get('cap','—')} USD; prioritize {', '.join(budget_cfg.get('prefer', []))}.")
    return "\n".join(lines)

def export_pdf(filename, ctx, recs, bundle, reasoning_text):
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        x, y = 2*cm, height - 2*cm

        c.setFont("Helvetica-Bold", 16)
        c.drawString(x, y, "EcoHome Advisor — Safe Report")
        y -= 1.0*cm

        c.setFont("Helvetica", 11)
        c.drawString(x, y, f"Climate zone: {ctx.get('climate_zone')}  Humidity: {ctx.get('humidity')}  Orientation: {ctx.get('orientation')}")
        y -= 0.6*cm
        c.drawString(x, y, f"Window-to-wall ratio: {ctx.get('window_area_ratio'):.2f}  Budget: {ctx.get('budget')}")
        y -= 0.6*cm
        c.drawString(x, y, f"Roof lightness: {ctx.get('roof_L'):.1f}  Roof color: {ctx.get('roof_color')}  Glass/reflection: {ctx.get('glass_ratio')*100:.1f}%")
        y -= 0.9*cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Recommendations")
        y -= 0.6*cm
        c.setFont("Helvetica", 11)
        if not recs:
            c.drawString(x, y, "None for current inputs.")
            y -= 0.6*cm
        else:
            for r in recs:
                line = f"- [{r['part']}] {', '.join(r['recommend'])}  (Saving: {r['saving_pct']})"
                for seg in [line[i:i+95] for i in range(0, len(line), 95)]:
                    c.drawString(x, y, seg)
                    y -= 0.5*cm
                    if y < 3*cm:
                        c.showPage(); y = height - 2*cm
                why = f"  Why: {r['rationale']}"
                for seg in [why[i:i+95] for i in range(0, len(why), 95)]:
                    c.drawString(x, y, seg)
                    y -= 0.5*cm
                    if y < 3*cm:
                        c.showPage(); y = height - 2*cm

        if bundle:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x, y, "Cost & Priority (rough)")
            y -= 0.6*cm
            c.setFont("Helvetica", 11)
            for it in bundle:
                cost_str = f"~${it['est_cost']}" if it["est_cost"] else "—"
                line = f"- [{it['part']}] {', '.join(it['recommend'])}  Cost: {cost_str}  Saving: {it['saving'][0]}–{it['saving'][1]}%  Priority: {it['priority']}"
                for seg in [line[i:i+95] for i in range(0, len(line), 95)]:
                    c.drawString(x, y, seg)
                    y -= 0.5*cm
                    if y < 3*cm:
                        c.showPage(); y = height - 2*cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "AI Reasoning")
        y -= 0.6*cm
        c.setFont("Helvetica", 11)
        for seg in [reasoning_text[i:i+95] for i in range(0, len(reasoning_text), 95)]:
            c.drawString(x, y, seg)
            y -= 0.5*cm
            if y < 3*cm:
                c.showPage(); y = height - 2*cm

        c.showPage()
        c.save()
        pdf_bytes = buffer.getvalue()
        buffer.close()
        with open(filename, "wb") as f:
            f.write(pdf_bytes)
        return filename
    except Exception as e:
        st.error(f"PDF export failed: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("Inputs (safe defaults)")
    climate_zone = st.selectbox("Climate zone", ["hot_humid", "temperate", "cold_dry"], index=1)
    humidity = st.selectbox("Humidity level", ["low", "medium", "high"], index=1)
    orientation = st.selectbox("Main facade orientation", ["north","south","east","west"], index=1)
    window_area_ratio = st.slider("Window-to-wall ratio (0 = very few windows; 0.6 = many windows)", 0.0, 0.6, 0.2, 0.01)
    budget = st.selectbox("Budget tier", ["low","mid","high"], index=1)
    rules_text = st.text_area("Rules (YAML, editable)", value=DEFAULT_RULES, height=260)

left, right = st.columns([1,1])

with left:
    st.subheader("① Upload exterior/roof photo (optional)")
    img_file = st.file_uploader("Choose JPG/PNG (you can skip this)", type=["jpg","jpeg","png"])
    img_bgr = None
    image = None
    if img_file is not None:
        try:
            image = Image.open(img_file)
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Downscale very large images to avoid memory errors
            max_side = 1600
            w, h = image.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                image = image.resize((int(w*scale), int(h*scale)))
            st.image(image, caption="Uploaded image (preview)", use_container_width=True)
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except UnidentifiedImageError:
            st.error("Could not read this file as an image. Please upload JPG or PNG.")
        except Exception as e:
            st.error(f"Image processing error: {e}")

    run = st.button("Analyze & Recommend", type="primary")

if run:
    rules = safe_parse_yaml(rules_text)

    roof_L = 180.0
    roof_color = "unknown"
    glass_ratio = 0.0
    if img_bgr is not None:
        try:
            roof_L = simple_roof_lightness_score(img_bgr)
            roof_color = "dark" if roof_L < 140 else "light"
            glass_ratio = estimate_glass_reflection_ratio(img_bgr)
        except Exception as e:
            st.warning(f"Image metrics fallback due to error: {e}")
            roof_L, roof_color, glass_ratio = 180.0, "unknown", 0.0

    ctx = dict(
        climate_zone = climate_zone,
        humidity = humidity,
        orientation = orientation,
        window_area_ratio = window_area_ratio,
        budget = budget,
        roof_L = roof_L,
        roof_color = roof_color,
        glass_ratio = glass_ratio
    )

    recs, budget_cfg = apply_rules(rules, ctx)

    with right:
        st.subheader("② Recommendation List")
        if not recs:
            st.info("No strong recommendations yet. Try uploading a photo or adjust humidity/orientation/window ratio—or edit the rules on the left.")
        else:
            for r in recs:
                box = st.container()   # no border arg for max compatibility
                with box:
                    st.markdown(f"**Part**: {r['part']}")
                    st.markdown(f"**Recommend**: {', '.join(r['recommend'])}")
                    st.markdown(f"**Estimated energy saving**: {r['saving_pct']}")
                    st.caption(f"Why: {r['rationale']}")

        st.markdown("---")
        st.subheader("③ Cost & Priority (rough)")
        bundle = simple_cost_and_priority(recs, budget_cfg)
        if bundle:
            for it in bundle:
                cost_str = f"${it['est_cost']}" if it['est_cost'] else "—"
                st.write(f"- [{it['part']}] {', '.join(it['recommend'])}  | Cost: {cost_str}  | Saving: {it['saving'][0]}–{it['saving'][1]}%  | Priority: {it['priority']}")
        else:
            st.write("—")

        st.markdown("---")
        st.subheader("④ AI Reasoning")
        reasoning_text = ai_reasoning(ctx, recs, budget_cfg)
        st.markdown(reasoning_text)

        st.markdown("---")
        st.subheader("⑤ Assumptions & Limits")
        st.caption("Heuristics only; photo cues are rough. Savings are ranges, not guaranteed. Costs are budget-cap splits for demo purposes.")

        if st.button("Export PDF report"):
            filename = "EcoHome_Report_Safe.pdf"
            path = export_pdf(filename, ctx, recs, bundle, reasoning_text)
            if path:
                with open(path, "rb") as f:
                    st.download_button("Download PDF", f.read(), file_name=filename, mime="application/pdf")
else:
    with right:
        st.info("Select inputs on the left, optionally upload a photo, then click **Analyze & Recommend**.")
