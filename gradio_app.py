import os
import gradio as gr
import requests
import json

# ---------------------------------------------------------------------------
# Configuration – reads from env var PREDICT_API_URL; falls back to default
# ---------------------------------------------------------------------------
API_URL = os.environ.get("PREDICT_API_URL", "http://localhost:8008/predict")


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_obesity(
    height: float,
    weight: float,
    age: float,
    vegetable_freq: float,
    meals_count: float,
    water_consumption: float,
    physical_activity: float,
    tech_time: float,
    family_history: str,
    high_calorie: str,
    alcohol: str,
):
    """Send patient data to the prediction API and return the result."""

    payload = {
        "Height": height,
        "Weight": weight,
        "Age": age,
        "Vegetable_consumption_frequency": vegetable_freq,
        "main_meals_count": meals_count,
        "water_consumption": water_consumption,
        "Physical_activity_frequency": physical_activity,
        "Tech_device_time": tech_time,
        "family_history_with_overweight": family_history,
        "High_calorie_food": high_calorie,
        "Alcohol_consumption": alcohol,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()

        prediction = result.get("prediction", "N/A")

        # Styled HTML card showing only the prediction
        html = (
            '<div class="result-card">'
            '  <div class="result-label">Obesity Classification</div>'
            f'  <div class="result-value">{prediction}</div>'
            '</div>'
        )
        return html

    except requests.exceptions.ConnectionError:
        return (
            '<div class="result-card result-error">'
            '  <div class="result-label">⚠️ Connection Error</div>'
            f'  <div class="result-value" style="color:#ef4444;font-size:1.2rem;">'
            f'Could not reach the API at {API_URL}</div>'
            '</div>'
        )
    except requests.exceptions.HTTPError as exc:
        return (
            '<div class="result-card result-error">'
            f'  <div class="result-label">⚠️ HTTP Error {exc.response.status_code}</div>'
            f'  <div class="result-value" style="color:#ef4444;font-size:1.2rem;">'
            f'{exc.response.text}</div>'
            '</div>'
        )
    except Exception as exc:
        return (
            '<div class="result-card result-error">'
            '  <div class="result-label">⚠️ Unexpected Error</div>'
            f'  <div class="result-value" style="color:#ef4444;font-size:1.2rem;">{exc}</div>'
            '</div>'
        )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Obesity Level Predictor",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        .main-header {
            text-align: center;
            padding: 1.5rem 0 0.5rem;
        }
        .main-header h1 {
            background: linear-gradient(135deg, #0d9488, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 800;
        }
        .main-header p {
            color: #94a3b8;
            font-size: 1rem;
        }
        footer { display: none !important; }

        /* ---------- Result card ---------- */
        .result-card {
            text-align: center;
            padding: 2rem 1.5rem;
            margin: 1rem auto;
            max-width: 520px;
            border-radius: 16px;
            background: linear-gradient(145deg, #0f172a, #1e293b);
            border: 1px solid rgba(16, 185, 129, 0.3);
            box-shadow: 0 0 28px rgba(16, 185, 129, 0.15),
                        0 4px 12px rgba(0, 0, 0, 0.4);
        }
        .result-label {
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 0.6rem;
        }
        .result-value {
            font-size: 2.6rem;
            font-weight: 900;
            color: #22c55e;
            text-shadow: 0 0 18px rgba(34, 197, 94, 0.45);
        }
        .result-error {
            border-color: rgba(239, 68, 68, 0.35);
            box-shadow: 0 0 28px rgba(239, 68, 68, 0.12),
                        0 4px 12px rgba(0, 0, 0, 0.4);
        }
        .result-placeholder {
            text-align: center;
            padding: 1.8rem;
            color: #64748b;
            font-style: italic;
            font-size: 1rem;
        }
    """,
) as demo:

    # Header
    gr.HTML(
        """
        <div class="main-header">
            <h1>🏥 Obesity Level Predictor</h1>
            <p>Enter your personal information and daily habits to predict your obesity classification.</p>
        </div>
        """
    )

    with gr.Row():
        # ---- Left column: Personal Information ----
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Personal Information")
            height = gr.Number(
                label="Height (m)",
                value=1.70,
                minimum=0.5,
                maximum=2.50,
                step=0.01,
                info="Your height in metres (e.g. 1.75)",
            )
            weight = gr.Number(
                label="Weight (kg)",
                value=70.0,
                minimum=10.0,
                maximum=300.0,
                step=0.5,
                info="Your weight in kilograms",
            )
            age = gr.Number(
                label="Age",
                value=25,
                minimum=1,
                maximum=120,
                step=1,
                info="Your age in years",
            )
            family_history = gr.Radio(
                label="Family history with overweight?",
                choices=["yes", "no"],
                value="no",
                info="Does your family have a history of overweight?",
            )

        # ---- Right column: Daily Behaviour ----
        with gr.Column(scale=1):
            gr.Markdown("### 🍎 Daily Behaviour")
            vegetable_freq = gr.Slider(
                label="Vegetable consumption frequency",
                minimum=1.0,
                maximum=3.0,
                step=0.5,
                value=2.0,
                info="1 = Never, 2 = Sometimes, 3 = Always",
            )
            meals_count = gr.Slider(
                label="Number of main meals per day",
                minimum=1.0,
                maximum=4.0,
                step=0.5,
                value=3.0,
                info="How many main meals do you eat per day?",
            )
            water_consumption = gr.Slider(
                label="Daily water consumption",
                minimum=1.0,
                maximum=3.0,
                step=0.5,
                value=2.0,
                info="1 = Less than 1L, 2 = 1–2L, 3 = More than 2L",
            )
            physical_activity = gr.Slider(
                label="Physical activity frequency",
                minimum=0.0,
                maximum=3.0,
                step=0.5,
                value=1.0,
                info="0 = None, 1 = 1–2 days, 2 = 2–4 days, 3 = 4–5 days",
            )
            tech_time = gr.Slider(
                label="Time using technology devices (hrs/day)",
                minimum=0.0,
                maximum=3.0,
                step=0.5,
                value=1.0,
                info="0 = 0–2h, 1 = 3–5h, 2 = 5+h",
            )
            high_calorie = gr.Radio(
                label="Frequent high-calorie food consumption?",
                choices=["yes", "no"],
                value="no",
                info="Do you frequently eat high-calorie food?",
            )
            alcohol = gr.Radio(
                label="Alcohol consumption",
                choices=["no", "Sometimes", "Frequently"],
                value="no",
                info="How often do you consume alcohol?",
            )

    # ---- Submit & Result ----
    with gr.Row():
        with gr.Column(scale=1):
            submit_btn = gr.Button(
                "🔍  Predict Obesity Level",
                variant="primary",
                size="lg",
            )
        with gr.Column(scale=1):
            clear_btn = gr.Button("🗑️  Clear", variant="secondary", size="lg")

    result_output = gr.HTML(
        value='<div class="result-placeholder">Your prediction result will appear here after you click <b>Predict</b>.</div>',
    )

    # Wire up the button
    submit_btn.click(
        fn=predict_obesity,
        inputs=[
            height,
            weight,
            age,
            vegetable_freq,
            meals_count,
            water_consumption,
            physical_activity,
            tech_time,
            family_history,
            high_calorie,
            alcohol,
        ],
        outputs=result_output,
    )

    # Clear button resets everything
    clear_btn.click(
        fn=lambda: (
            1.70, 70.0, 25, 2.0, 3.0, 2.0, 1.0, 1.0,
            "no", "no", "no",
            '<div class="result-placeholder">Your prediction result will appear here after you click <b>Predict</b>.</div>',
        ),
        inputs=[],
        outputs=[
            height, weight, age,
            vegetable_freq, meals_count, water_consumption,
            physical_activity, tech_time,
            family_history, high_calorie, alcohol,
            result_output,
        ],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
