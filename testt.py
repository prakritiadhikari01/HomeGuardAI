import cv2
from app.services.recognition_service import RecognitionService


def draw_results(frame, matches):
    print(f"[DEBUG] Matches: {len(matches)}")

    for i, match in enumerate(matches):

        label = "Unknown"
        color = (0, 0, 255)

        if match.get("matched"):
            user = match.get("user", {})
            score = match.get("score", 0)

            label = f"{user.get('label_name', 'Unknown')} ({score:.2f})"
            color = (0, 255, 0)

            print(f"[MATCH FOUND] {label}")

        else:
            print("[INFO] Unknown Face")

        cv2.putText(
            frame,
            label,
            (50, 50 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return frame


def main():

    print("=" * 50)
    print("Starting HomeGuard AI Recognition Test")
    print("=" * 50)

    service = RecognitionService()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam not opened")
        return

    frame_count = 0

    while True:

        frame_count += 1

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        print(f"\n[FRAME {frame_count}] Running recognition...")

        matches = service.recognize_frame(frame)

        frame = draw_results(frame, matches)

        cv2.imshow("HomeGuard AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()