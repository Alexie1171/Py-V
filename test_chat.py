from inference.engine.chat import ChatEngine


def main():

    chat = ChatEngine()

    session_id = "test_session_1"

    print("PY-V Phase 7 Test Started")
    print("Type 'exit' to stop\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        result = chat.chat(session_id, user_input)

        print("\nPY-V:")
        print(result["response"])
        print(f"\n[mode={result['mode']} | confidence={result['confidence']}]\n")


if __name__ == "__main__":
    main()