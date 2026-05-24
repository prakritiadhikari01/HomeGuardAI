import cv2

from app.services.recognition_service import RecognitionService


def draw_results(frame, faces, matches):
    """
    Draw recognition results on screen
    """

    print(f"[DEBUG] Total matches received: {len(matches)}")

    for i, match in enumerate(matches):

        print(f"[DEBUG] Match Data {i}: {match}")

        label = "Unknown"
        color = (0, 0, 255)

        # If face matched
        if match.get("matched"):

            user = match.get("user", {})
            score = match.get("score", 0)

            label = f"{user.get('name', 'Unknown')} ({score:.2f})"
            color = (0, 255, 0)

            print(f"[MATCH FOUND] {label}")

        else:
            print("[INFO] Unknown Face Detected")

        # Draw text on frame
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

    # STEP 1 — Initialize Recognition Service
    try:
        print("[STEP 1] Initializing Recognition Service...")

        service = RecognitionService()

        print("[SUCCESS] Recognition Service Initialized")

    except Exception as e:
        print("[ERROR] Failed to initialize RecognitionService")
        print("ERROR:", e)
        return

    # STEP 2 — Open Webcam
    print("\n[STEP 2] Opening Webcam...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[SUCCESS] Webcam Opened Successfully")

    frame_count = 0

    # STEP 3 — Start Loop
    while True:

        frame_count += 1

        print(f"\n[FRAME {frame_count}] Capturing Frame...")

        ret, frame = cap.read()

        # Check webcam frame
        if not ret:
            print("[ERROR] ret = False (camera read failed)")
            continue

        if frame is None:
            print("[ERROR] Frame is None")
            continue

        print("[SUCCESS] Frame Captured")

        # Print frame info
        try:
            print(f"[DEBUG] Frame Shape: {frame.shape}")
        except Exception as e:
            print("[ERROR] Could not get frame shape:", e)

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)

        # STEP 4 — Recognition
        try:
            print("[STEP 4] Running Recognition Pipeline...")

            matches = service.recognize_frame(frame)

            print("[SUCCESS] Recognition Completed")

        except Exception as e:
            print("[ERROR] Recognition Pipeline Failed")
            print("ERROR:", e)
            continue

        # STEP 5 — Draw Results
        try:
            print("[STEP 5] Drawing Results...")

            frame = draw_results(frame, None, matches)

            print("[SUCCESS] Results Drawn")

        except Exception as e:
            print("[ERROR] Drawing Results Failed")
            print("ERROR:", e)

        # STEP 6 — Show Window
        try:
            cv2.imshow("HomeGuard AI - Face Recognition", frame)

        except Exception as e:
            print("[ERROR] Failed to show window")
            print("ERROR:", e)

        # Press Q to Quit
        key = cv2.waitKey(1)

        if key == ord("q"):
            print("\n[INFO] Q Pressed - Exiting Program")
            break

    # Cleanup
    print("\n[INFO] Releasing Webcam...")

    cap.release()

    print("[INFO] Destroying Windows...")

    cv2.destroyAllWindows()

    print("[SUCCESS] Program Closed Successfully")


if __name__ == "__main__":
    main()