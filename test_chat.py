from datetime import datetime

from inference.engine.chat import ChatEngine


def main():

    chat = ChatEngine()

    # A new chat each run: V sees this chat's last turns, and long-term
    # memory (facts) carries over from all chats
    session_id = f"terminal-{datetime.now():%Y%m%d-%H%M%S}"

    print("V is ready")
    print("Type 'exit' to stop\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        result = chat.chat(session_id, user_input)

        print("\nV:")
        print(result["response"])
        if result.get("sources"):   # a web lookup (Phase 13)
            print("\nSources: " + " | ".join(f"{s['title']} <{s['url']}>" for s in result["sources"]))
        if result.get("note"):
            print(f"\n({result['note']})")
        study = result.get("study")
        if study and study.get("running"):
            print(f"\n(studying {study['topic']}: {study['minutes_left']:g} min left, {study['notes']} notes)")

        rag_info = f" | rag_chunks={result['rag_chunks']}" if result["rag_chunks"] > 0 else ""
        notes_info = f" | notes={result['notes']}" if result.get("notes") else ""
        load_info = f" | laptop={result['load']}" if result.get("load") else ""
        print(f"\n[mode={result['mode']} | confidence={result['confidence']}{rag_info}{notes_info}{load_info}]\n")


if __name__ == "__main__":
    main()
