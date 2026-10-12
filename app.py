import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "customer_service_intent_model.joblib"
UI_PATH = PROJECT_DIR / "index.html"
HOST = "127.0.0.1"
PORT = 8000
MIN_CONFIDENCE = 0.35
MAX_MESSAGE_LENGTH = 1000

RESPONSES = {
    "cancel_order": "I can help with an order cancellation. I can't access or change orders here, but customer support can check whether your order is still eligible. Please have your order number ready.",
    "change_order": "I can help with changing an order. Changes depend on its current status, so please contact customer support with your order number.",
    "change_shipping_address": "For a shipping address change, contact customer support as soon as possible and provide your order number. I can't view or update shipping details here.",
    "check_cancellation_fee": "Cancellation fees can depend on the order and its status. Please check your order terms or ask customer support with your order number.",
    "check_invoice": "I can point you in the right direction for an invoice. Please sign in to your account and check your orders, or contact customer support if you can't find it.",
    "check_payment_methods": "Available payment methods can vary by location and checkout. Please check the payment options shown at checkout or contact customer support.",
    "check_refund_policy": "Refund eligibility and timing depend on the purchase and the store's policy. Please review the refund policy for your order or contact customer support.",
    "complaint": "I'm sorry you're having a frustrating experience. Please tell customer support what happened and include any relevant order number so they can look into it.",
    "contact_customer_service": "You can reach the customer service team through the support or contact page. Include your order number if your question is about an order.",
    "contact_human_agent": "This request may need a support agent. Please use the customer service contact page and include any relevant order number.",
    "create_account": "To create an account, choose the sign-up or register option on the service's website and follow the instructions.",
    "delete_account": "For account deletion, contact customer support or use the account privacy settings. They can confirm the steps and any consequences before deletion.",
    "delivery_options": "Delivery options and prices are normally shown at checkout after you enter your delivery address.",
    "delivery_period": "Delivery times depend on the shipping option and destination. Check the estimate in your order confirmation or tracking page.",
    "edit_account": "You can usually update your account details from the profile or account settings page after signing in.",
    "get_invoice": "Try signing in and opening your order history to download the invoice. If it isn't available, ask customer support for help.",
    "get_refund": "To ask about a refund, check the order's refund or return options, or contact customer support with your order number. I can't access payment records here.",
    "newsletter_subscription": "You can usually subscribe or unsubscribe using the newsletter option on the website or the link in a newsletter email.",
    "payment_issue": "Please check that your payment details are correct and contact your bank if needed. For a declined or duplicate charge, contact customer support; don't share full card details in chat.",
    "place_order": "To place an order, add items to your cart and follow the checkout steps. Review the delivery address and total before confirming.",
    "recover_password": "Use the 'Forgot password' link on the sign-in page and follow the password-reset instructions sent to your registered email.",
    "registration_problems": "If registration isn't working, check that the required fields are complete and try again. If the issue continues, contact customer support.",
    "review": "You can usually leave a review from the product page or your order history. Please follow the site's review guidelines.",
    "set_up_shipping_address": "You can usually add a delivery address in your account settings or during checkout. Check the address carefully before placing an order.",
    "switch_account": "Sign out of the current account, then sign in with the other account's email address. If you can't access it, use account recovery.",
    "track_order": "You can check the latest delivery status using the tracking link in your shipping confirmation or in your account's order history.",
    "track_refund": "Refund status may be available in your order history. Processing times vary; contact customer support with your order number if it seems delayed.",
}


class ChatHandler(BaseHTTPRequestHandler):
    model: Any

    def send_json(self, status: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if urlparse(self.path).path not in ("/", "/index.html"):
            self.send_error(404, "Page not found")
            return
        try:
            page = UI_PATH.read_bytes()
        except OSError as error:
            self.send_error(500, f"Could not read the chat interface: {error}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404, "Endpoint not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_MESSAGE_LENGTH * 4:
                self.send_json(400, {"error": "Please send a shorter message."})
                return
            body = json.loads(self.rfile.read(content_length))
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": "The request must contain valid JSON."})
            return

        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, str) or not message.strip():
            self.send_json(400, {"error": "Please enter a question."})
            return
        message = message.strip()
        if len(message) > MAX_MESSAGE_LENGTH:
            self.send_json(400, {"error": "Please keep your question under 1,000 characters."})
            return

        probabilities = self.model.predict_proba([message])[0]
        best_index = int(probabilities.argmax())
        confidence = float(probabilities[best_index])
        intent = str(self.model.classes_[best_index])
        if confidence < MIN_CONFIDENCE:
            answer = (
                "I'm not sure I understood that. Could you rephrase your question "
                "or contact customer support for help?"
            )
        else:
            answer = RESPONSES.get(
                intent,
                "I may be able to point you in the right direction. Please contact "
                "customer support with a little more detail.",
            )
        self.send_json(
            200,
            {
                "answer": answer,
                "intent": intent,
                "confidence": confidence,
            },
        )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"Trained model not found at {MODEL_PATH}. Run train.py first."
        )
    if not UI_PATH.is_file():
        raise SystemExit(f"Chat interface not found at {UI_PATH}.")

    ChatHandler.model = joblib.load(MODEL_PATH)
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"Customer service chat is running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping customer service chat.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
