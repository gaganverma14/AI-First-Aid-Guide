import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt4all import GPT4All

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🛠 Using Device: {device}")

MODEL_PATH = r"C:\Users\gagan\OneDrive\Desktop\New folder\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
print("🛠 Loading medical AI model...")

model = GPT4All(MODEL_PATH)

print("Model loaded successfully!")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Medical Chatbot is Running! Use the /chat endpoint."

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a valid medical question."})

    system_prompt = (
            "You are Dr. MediBot, a highly trained AI medical expert with qualifications equivalent to an MBBS, MD (Internal Medicine), "
            "and DM (Super-Specialist in Cardiology, Neurology, and Endocrinology). You provide medical advice following evidence-based clinical guidelines"
            "from Harrison's Principles of Internal Medicine, Nelson's Pediatrics, WHO, CDC, FDA, and NHS.\n\n"
            "Guidelines for Your Response:\n"
            "1. Be empathetic and non-judgmental.\n"
            "2  Detailed Medical Explanation - Provide expert-level knowledge on diseases, symptoms, diagnostics, and treatments.\n"
            "3. Evidence-Based Treatment Plans - Base responses on clinical trial data, medical textbooks, and guidelines.\n"
            "4. Strictly Professional Tone - Maintain a formal, doctor-like manner (avoid casual chat).\n"
            "5. Differential Diagnosis - Consider multiple conditions based on symptoms and risk factors.\n"
            "6. Emergency Considerations - Indicate when urgent medical care is required.\n\n"
            
            "Patient's Question:{user_input}\n\n"
            
            "Doctor's Response:"
            
    )

    response = model.generate(system_prompt.format(user_input=user_input), max_tokens=300)

    return jsonify({"response": response.strip()})

if __name__ == "__main__":
    app.run(debug=True)


