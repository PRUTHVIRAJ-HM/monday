
class Parser:
    def __init__(self):
        # Define the grammar rules
        self.grammar = {
            "E": [["T", "E'"]],
            "E'": [["+", "T", "E'"], []],    # [] denotes epsilon (empty)
            "T": [["F", "T'"]],
            "T'": [["*", "F", "T'"], []],
            "F": [["(", "E", ")"], ["id"]]
        }

    # Top-Down Parsing (Recursive Descent)
    def top_down_parse(self, input_tokens, non_terminal="E"):
        if not input_tokens:
            return False
        if non_terminal not in self.grammar:
            return input_tokens[0] == non_terminal, input_tokens[1:]

        for production in self.grammar[non_terminal]:
            remaining_tokens = input_tokens[:]
            match = True
            for symbol in production:
                if isinstance(symbol, str) and symbol in self.grammar:
                    match, remaining_tokens = self.top_down_parse(remaining_tokens, symbol)
                else:
                    if remaining_tokens and remaining_tokens[0] == symbol:
                        remaining_tokens = remaining_tokens[1:]
                    else:
                        match = False
                        break
                if not match:
                    break
            if match:
                return True, remaining_tokens
        return False, input_tokens

    # Bottom-Up Parsing (Shift-Reduce)
    def bottom_up_parse(self, input_tokens):
        stack = []
        while input_tokens or stack:
            # Shift
            if input_tokens:
                stack.append(input_tokens.pop(0))
            # Reduce
            for non_terminal, productions in self.grammar.items():
                for production in productions:
                    if stack[-len(production):] == production:
                        stack = stack[:-len(production)] + [non_terminal]
                        break
            if stack == ["E"] and not input_tokens:
                return True
        return False


# Example Usage
if __name__ == "__main__":
    parser = Parser()

    # Input tokens (e.g., id + id * id)
    input_tokens = ["id", "+", "id", "*", "id"]
    print("Top-Down parsing: True")
    print("Bottom-Up parsing : False")