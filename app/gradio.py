import io, json, httpx, gradio as gr
from PIL import Image
from src.v1.config import API_URL

# function for connecting to the server
def api_call(endpoint: str, method: str = "GET", token: str = "", json_data: dict = None, raw_response: bool = False):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        res = httpx.request(method, f"{API_URL}{endpoint}", json=json_data, headers=headers, timeout=30.0)
        if raw_response:
            return Image.open(io.BytesIO(res.content)) if res.status_code == 200 else None
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# authentication function
def handle_auth(action, user, pwd):
    res = api_call(f"/{action.lower()}", "POST", json_data={"username": user, "password": pwd})
    token = res.get("access_token", "") if action == "Login" and isinstance(res, dict) else ""
    return token, res


# for /predict and /explain as both of them take single dataset
def handle_single_inference(action, token, name, age, gender, blood_type, condition, date_adm, doctor, hospital, insurance, billing, room_num, admission, date_dis, medication):
    payload = {
        "data": {
            "Name": name,
            "Age": int(age),
            "Gender": gender,
            "Blood Type": blood_type,
            "Medical Condition": condition,
            "Date of Admission": date_adm,
            "Doctor": doctor,
            "Hospital": hospital,
            "Insurance Provider": insurance,
            "Billing Amount": float(billing),
            "Room Number": int(room_num),
            "Admission Type": admission,
            "Discharge Date": date_dis,
            "Medication": medication
        }
    }
    
    # if /predict
    if action == "Predict":
        return api_call("/predict", "POST", token=token, json_data=payload), None
    
    # if not
    img = api_call("/explain", "POST", token=token, json_data=payload, raw_response=True)
    
    return {"status": "image_generated" if img else "failed"}, img


# for handling batch inference
def handle_batch_inference(token, json_str):
    try:
        data_list = json.loads(json_str)
        payload = {"data": data_list} if isinstance(data_list, list) else data_list
        return api_call("/predict/batch", "POST", token=token, json_data=payload)
    except Exception as e:
        return {"error": f"Invalid JSON format: {str(e)}"}


# simple UI layout
with gr.Blocks(title="Healthcare Test Results Platform") as demo:
    
    auth_token = gr.State("")

    gr.Markdown("Healthcare Test Results ML Control Center")
    
    with gr.Tab("Inference & Authentication"):
        with gr.Row():
            # Auth & System Status
            with gr.Column(scale=1):
                gr.Markdown("Auth & System")
                user_in = gr.Textbox(label="Username")
                pass_in = gr.Textbox(label="Password", type="password")
                
                with gr.Row():
                    btn_login = gr.Button("Login", variant="primary")
                    btn_reg = gr.Button("Register")
                
                with gr.Row():
                    btn_status = gr.Button("/status")
                    btn_health = gr.Button("/health")
                    
                token_disp = gr.Textbox(label="Active Bearer Token", interactive=False, max_lines=1)
                sys_out = gr.JSON(label="System Response")

            # for handling single inference and SHAP
            with gr.Column(scale=2):
                gr.Markdown("Single Patient Inference & Explanation")
                with gr.Row():
                    name = gr.Textbox(label="Patient Name", value="Bobby JacksOn")
                    age = gr.Slider(18, 100, value=30, label="Age")
                    gender = gr.Radio(["Female", "Male"], value="Male", label="Gender")
                    blood_type = gr.Dropdown(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], value="B-", label="Blood Type")
                with gr.Row():
                    condition = gr.Dropdown(["Cancer", "Obesity", "Diabetes", "Asthma", "Hypertension", "Arthritis"], value="Cancer", label="Medical Condition")
                    date_adm = gr.Textbox(label="Date of Admission", value="2024-01-31")
                    doctor = gr.Textbox(label="Doctor Name", value="Matthew Smith")
                    hospital = gr.Textbox(label="Hospital", value="Sons and Miller")
                with gr.Row():
                    insurance = gr.Dropdown(["Blue Cross", "Medicare", "Aetna", "Cigna", "UnitedHealthcare"], value="Blue Cross", label="Insurance Provider")
                    billing = gr.Number(label="Billing Amount", value=18856.28)
                    room_num = gr.Number(label="Room Number", value=328)
                with gr.Row():
                    admission = gr.Dropdown(["Urgent", "Emergency", "Elective"], value="Urgent", label="Admission Type")
                    date_dis = gr.Textbox(label="Discharge Date", value="2024-02-02")
                    medication = gr.Dropdown(["Paracetamol", "Ibuprofen", "Aspirin", "Penicillin", "Lipitor"], value="Paracetamol", label="Medication")

                with gr.Row():
                    btn_pred = gr.Button("Predict (/predict)", variant="primary")
                    btn_exp = gr.Button("Explain (/explain)")

                pred_out = gr.JSON(label="Prediction Result")
                img_out = gr.Image(label="SHAP Explanation Plot", type="pil")

    with gr.Tab("Model Metrics & Evaluation"):
        gr.Markdown("Model Performance Analysis (/evaluate)")
        btn_eval = gr.Button("Fetch Evaluation Plots", variant="primary")
        eval_img_out = gr.Image(label="Performance Charts (Bar Chart, ROC, PR Curves)", type="pil")

    with gr.Tab("Batch Operations & Monitoring"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Batch Processing (/predict/batch)")
                batch_json = gr.TextArea(
                    label="Batch JSON Input",
                    value='[\n  {\n    "Name": "Bobby JacksOn", "Age": 30, "Gender": "Male", "Blood Type": "B-",\n    "Medical Condition": "Cancer", "Date of Admission": "2024-01-31", "Doctor": "Matthew Smith",\n    "Hospital": "Sons and Miller", "Insurance Provider": "Blue Cross", "Billing Amount": 18856.28,\n    "Room Number": 328, "Admission Type": "Urgent", "Discharge Date": "2024-02-02", "Medication": "Paracetamol"\n  }\n]',
                    lines=8
                )
                btn_batch = gr.Button("Run Batch Prediction")
                batch_out = gr.JSON(label="Batch Results")

            with gr.Column():
                gr.Markdown("### Ground Truth & Production Monitoring")
                with gr.Row():
                    req_id = gr.Textbox(label="Request ID")
                    # mapping categorical test results targets to integers matching your updated pipelines (0, 1, 2)
                    test_results_val = gr.Radio([0, 1, 2], value=0, label="Actual Results (0: Normal, 1: Abnormal, 2: Inconclusive)")
                btn_label = gr.Button("Submit Ground Truth (/label)")
                label_out = gr.JSON(label="Label Response")

                gr.Markdown("#### Data Drift & Retraining Check")
                btn_monitor = gr.Button("Run Production Monitoring (/monitor)", variant="primary")
                mon_out = gr.JSON(label="Monitoring Diagnostics")

    # for handling events
    btn_login.click(lambda u, p: handle_auth("Login", u, p), inputs=[user_in, pass_in], outputs=[auth_token, sys_out])
    btn_reg.click(lambda u, p: handle_auth("Register", u, p), inputs=[user_in, pass_in], outputs=[auth_token, sys_out])
    auth_token.change(lambda t: t, inputs=[auth_token], outputs=[token_disp])

    
    btn_status.click(lambda: api_call("/status"), outputs=[sys_out])
    btn_health.click(lambda t: api_call("/health", token=t), inputs=[auth_token], outputs=[sys_out])

    
    inf_inputs = [auth_token, name, age, gender, blood_type, condition, date_adm, doctor, hospital, insurance, billing, room_num, admission, date_dis, medication]
    btn_pred.click(lambda *args: handle_single_inference("Predict", *args), inputs=inf_inputs, outputs=[pred_out, img_out])
    btn_exp.click(lambda *args: handle_single_inference("Explain", *args), inputs=inf_inputs, outputs=[pred_out, img_out])

    
    btn_eval.click(lambda t: api_call("/evaluate", "GET", token=t, raw_response=True), inputs=[auth_token], outputs=[eval_img_out])
    btn_batch.click(handle_batch_inference, inputs=[auth_token, batch_json], outputs=[batch_out])
    btn_label.click(lambda t, r, c: api_call("/label", "POST", token=t, json_data={"request_id": r, "test_results": int(c)}), inputs=[auth_token, req_id, test_results_val], outputs=[label_out])
    btn_monitor.click(lambda t: api_call("/monitor", "POST", token=t), inputs=[auth_token], outputs=[mon_out])
    
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())