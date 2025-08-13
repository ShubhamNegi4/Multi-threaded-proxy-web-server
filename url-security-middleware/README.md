# 🧠 url-security-middleware

**Real-Time URL Risk Analysis Middleware for Proxy Servers using Machine Learning + Rule-Based Detection**

> A modular, Python-powered URL classification and validation system designed for seamless integration into proxy infrastructures. It inspects, scores, and explains the threat level of URLs in real time — acting as an intelligent middleware between clients and servers.

---

## 🧑‍💻 Contributors

gaurav bhatt  
shashank sharma  
shubham negi

---

## 🔐 Features

- ✅ **Multi-class URL Risk Scoring:** Detects phishing, malware, defacement, edge-case, and invalid URLs (`not_a_url`).
- 🧠 **CNN-LSTM Deep Learning Model:** Robust, explainable, and trained on real + synthetic data for high generalization.
- 🔄 **API and CLI:** REST API via FastAPI (`main.py`) and CLI demo (`predict_url.py`).
- 🧩 **Hybrid Detection:** Uses ML, allowlist, and pattern-based checks for maximum coverage.
- 📊 **Explainable Output:** Shows top-2 class probabilities and explanations for each prediction.
- 🗃️ **Logging:** All scans can be logged to a SQLite database (`url_logs.db`).
- 🏗️ **Synthetic Data Generation:** Generates realistic and adversarial URLs for all classes, including edge cases and junk.
- 📈 **Evaluation:** Outputs classification report and confusion matrix (PNG).
- 🧪 **Automated Testing:** Includes scripts for API and prediction testing.

---

## ⚙️ Architecture

```text
Client ⇄ Proxy (C/Node.js/Python)
            ⇓
      URL Middleware (Python)
         ↳ validate_url()
         ↳ Return JSON response
            ⇓
   Decision: Allow / Block / Alert
