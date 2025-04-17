from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO
from twilio.rest import Client
import cv2
import numpy as np
import torch
import joblib
import psycopg2
import os
import traceback
from PIL import Image

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173","https://frontend-mtkt.onrender.com"])
socketio = SocketIO(app, cors_allowed_origins="*")


TWILIO_SID = "AC7a0b089b8bd0d885834d455544c0ffff"
TWILIO_AUTH_TOKEN = "d0ba0087f8953783c60f7cef5af5a628"
TWILIO_PHONE_NUMBER = "+16893146301"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# ✅ Load YOLOv3 Model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("yolov3.txt").read().strip().split("\n")

# ✅ Load CNN Model
lb = joblib.load("lb.pkl")

class CustomCNN(torch.nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.conv4 = torch.nn.Conv2d(64, 128, 5)
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, len(lb.classes_))
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = self.pool(torch.nn.functional.relu(self.conv4(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Load CNN Model
model = CustomCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ✅ Connect to PostgreSQL
# ✅ Connect to PostgreSQL using DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL, sslmode='require')  # sslmode is important for Render
conn.autocommit = True

# ✅ Function to Stream Video from Database & Detect
def generate_frames(video_id):
    try:
        print(f"📡 Fetching video with ID: {video_id}")

        cursor = conn.cursor()
        cursor.execute("SELECT filedata FROM videos WHERE id = %s", (video_id,))
        row = cursor.fetchone()
        cursor.close()

        if not row:
            print(f"❌ ERROR: Video ID {video_id} NOT FOUND in Database")
            return iter([])

        # ✅ Save video as temp file
        video_path = f"temp_video_{video_id}.mp4"
        with open(video_path, "wb") as f:
            f.write(row[0])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ ERROR: OpenCV cannot open the video.")
            return iter([])

        frame_count = 0
        drowning_alert_sent = False  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("🔴 End of video reached.")
                break

            frame_count += 1
            height, width, _ = frame.shape

            # ✅ YOLOv3 Object Detection
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes, confidences, class_ids = [], [], []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and classes[class_id] == "person":
                        x, y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if indices is not None and len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    color = (0, 255, 0)  
                    label = "SAFE"

                    if frame_count % 5 == 0:
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 224))
                        pil_image = np.transpose(np.array(pil_image), (2, 0, 1)).astype(np.float32)
                        pil_image = torch.tensor(pil_image, dtype=torch.float).unsqueeze(0)

                        with torch.no_grad():
                            outputs = model(pil_image)
                            _, preds = torch.max(outputs, 1)
                            label_pred = lb.classes_[preds.item()]

                            if label_pred == "drowning":
                                color = (0, 0, 255)
                                label = "DROWNING"

                                if not drowning_alert_sent:
                                    drowning_alert_sent = True
                                    print("🚨 Drowning Detected! Sending WebSocket alert...")
                                    socketio.emit("drowningAlert", {"videoId": video_id})

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        cap.release()
        os.remove(video_path)  # ✅ Remove temp video file after processing
        print("✅ Video processing completed.")

    except Exception as e:
        print(f"❌ ERROR in generate_frames: {str(e)}")
        traceback.print_exc()
        return iter([])
    
@app.route("/alerts", methods=["POST"])
@app.route("/alerts", methods=["POST"])
def send_alert():
    try:
        data = request.get_json()
        video_id = data.get("videoId")
        supervisor_id = data.get("supervisorId")

        if not video_id or not supervisor_id:
            return jsonify({"error": "❌ Missing video ID or supervisor ID!"}), 400

        cursor = conn.cursor()

        # ✅ Fetch all lifeguard IDs
        cursor.execute("SELECT id FROM lifeguard")
        lifeguard_ids = cursor.fetchall()

        if not lifeguard_ids:
            return jsonify({"error": "❌ No lifeguards found"}), 500

        for lifeguard_id in lifeguard_ids:
            try:
                # ✅ Insert Alert into alert_logs for each lifeguard
                cursor.execute(
                    """
                    INSERT INTO alert_logs (supervisor_id, video_id, lifeguard_id)
                    VALUES (%s, %s, %s)
                    """,
                    (supervisor_id, video_id, lifeguard_id[0])
                )
                conn.commit()

            except Exception as db_error:
                print(f"❌ ERROR inserting alert for lifeguard {lifeguard_id}: {db_error}")
                conn.rollback()
                return jsonify({"error": "❌ Database error while storing alert"}), 500

        cursor.close()

        # ✅ Send WebSocket Alert to Update Logs in Real-Time
        socketio.emit("lifeguardAlert", {"videoId": video_id})
        socketio.emit("updateAlertLogs")  # ✅ Notify frontend to refresh logs

        # ✅ Send SMS to ALL Lifeguards
        send_sms_to_lifeguards(video_id)

        return jsonify({"message": "✅ Alert sent successfully!"})

    except Exception as e:
        print("❌ ERROR in /alerts:", str(e))
        return jsonify({"error": "❌ Failed to send alert!"}), 500


@app.route("/alerts/latest", methods=["GET"])
def get_latest_alert():
    cursor = conn.cursor()
    cursor.execute(
        """SELECT video_id FROM alert_logs 
           ORDER BY timestamp DESC 
           LIMIT 1"""
    )
    latest_alert = cursor.fetchone()
    cursor.close()

    if latest_alert:
        return jsonify({"video_id": latest_alert[0]})
    else:
        return jsonify({"video_id": None})  # ✅ Return None instead of 404


# ✅ Route to Start Detection
@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "videoId" not in data:
        return jsonify({"error": "❌ No video ID provided"}), 400
    return jsonify({"stream_url": f"http://127.0.0.1:5001/detect-stream/{data['videoId']}"})

def send_sms_to_lifeguards(video_id):
    cursor = conn.cursor()

    # ✅ Fetch lifeguard phone numbers
    cursor.execute("SELECT id, phone_number FROM lifeguard")
    lifeguards = cursor.fetchall()

    message_text = f"🚨 Alert detected! Watch the video on your page. Video ID: {video_id}"

    for lifeguard_id, phone_number in lifeguards:
        formatted_phone = f"+91{phone_number}"  # ✅ Ensure country code for India

        try:
            client.messages.create(
                body=message_text,
                from_=TWILIO_PHONE_NUMBER,
                to=formatted_phone
            )

            print(f"📩 SMS Sent to Lifeguard {lifeguard_id} at {formatted_phone}")

        except Exception as e:
            print(f"❌ Failed to send SMS to {formatted_phone}: {str(e)}")

    cursor.close()

@app.route("/alerts", methods=["GET"])
def get_alert_logs():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alert_logs ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    cursor.close()

    return jsonify([
        {
            "id": log[0],
            "timestamp": log[1],
            "supervisor_id": log[2],
            "video_id": log[3],
            "lifeguard_id": log[4]
        } for log in logs
    ])



# ✅ Route to Stream Processed Video
@app.route("/detect-stream/<video_id>", methods=["GET"])
def detect_stream(video_id):
    return Response(generate_frames(video_id), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/lifeguard-video/<video_id>", methods=["GET"])
def lifeguard_video(video_id):
    return Response(generate_frames(video_id), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("🚀 Starting Flask with WebSockets on port 5001...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
