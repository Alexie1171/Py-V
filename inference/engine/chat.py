from inference.engine.controller import Controller
from inference.engine.context_manager import ContextManager
from inference.engine.prompt_builder import build_prompt
from inference.engine.model_loader import load_lora_model
from inference.engine.generator import generate_from_prompt


class ChatEngine:

    def __init__(self):
        self.controller = Controller()
        self.context_manager = ContextManager()

        self.model, self.tokenizer = load_lora_model()

    def chat(self, session_id: str, user_input: str):

        context = self.context_manager.load(session_id)
        clean_context = context.to_dict()

        intent = self.controller.detect_mode(user_input)

        prompt = build_prompt(
            mode=intent.mode,
            user_input=user_input,
            context=context.to_dict()
        )

        response = generate_from_prompt(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            mode=intent.mode,
            temperature=0.2
        )

        self.context_manager.append_history(
            context,
            user_input,
            response
        )

        self.context_manager.update(
            context,
            mode=intent.mode
        )

        return {
            "response": response,
            "mode": intent.mode,
            "confidence": intent.confidence,
            "flags": intent.flags
        }