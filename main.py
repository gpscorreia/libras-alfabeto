import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Não foi possível capturar o frame.")
            break

        frame = cv2.flip(frame, 1)

        cv2.imshow("Alfabeto em Libras", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()