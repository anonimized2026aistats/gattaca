"""
eval.py
Simple expression evaluator. Heavily based on original script by https://github.com/AndriiMysko and used with permission from the author.
"""

import enum
import re
from collections import deque


class TokenType(enum.Enum):
    """Expression token type enum"""
    SYMBOL = 0
    LOGIC_AND = 1
    LOGIC_OR = 2
    LOGIC_NOT = 3
    L_PARANTHESIS = 4
    R_PARANTHESIS = 5
    LOGIC_HIGH = 6
    LOGIC_LOW = 7


#: Expression token dictionary to simplify code a bit
TokenDic = {'and': TokenType.LOGIC_AND, 'or': TokenType.LOGIC_OR,
            'not': TokenType.LOGIC_NOT, '(': TokenType.L_PARANTHESIS, 
            ')': TokenType.R_PARANTHESIS, 'True': TokenType.LOGIC_HIGH, 'False': TokenType.LOGIC_LOW}

TokenDicRev = {value: key for key, value in TokenDic.items()}

token_cache = {}

#: Precedence value dictionary for postfix conversion
Precedence = {TokenType.LOGIC_NOT: 20, TokenType.LOGIC_AND: 11,
              TokenType.LOGIC_OR: 10, TokenType.SYMBOL: 0,
              TokenType.L_PARANTHESIS: 0, TokenType.R_PARANTHESIS: 0}


class ExpressionToken:
    """Expression token representation"""
    type = -1
    value = ""

    def __init__(self, type, value):
        self.type = type
        self.value = value


class LogicExpressionEvaluator:
    #: Dictionary for symbol evaluation
    dictionary = {}

    def __init__(self, role_dict: dict):
        self.dictionary = role_dict

    @classmethod
    def _tokenize(cls, id, in_str):
        """Tokenize the input string"""
        tokens = []

        if id in token_cache:
            return token_cache[id]

        print(in_str)

        # Remove whitespace and do basic tokenization
        in_str: list[str] = in_str.split()

        # Separate some operators / tokens away from symbols
        # TODO Logic NOT / AND / OR, also splitting into multiple symbols
        i = 0
        while i in range(len(in_str)):
            l_parenthesis_count = in_str[i].count(TokenDicRev[TokenType.L_PARANTHESIS])
            r_parenthesis_count = in_str[i].count(TokenDicRev[TokenType.R_PARANTHESIS])
            if l_parenthesis_count > 0 or r_parenthesis_count > 0:
                new = in_str[i].strip(TokenDicRev[TokenType.L_PARANTHESIS] + TokenDicRev[TokenType.R_PARANTHESIS])
                if len(new) > 0:
                    in_str[i] = new
                    if l_parenthesis_count:
                        for _ in range(l_parenthesis_count):
                            in_str.insert(i, TokenDicRev[TokenType.L_PARANTHESIS])
                    else:
                        for _ in range(r_parenthesis_count):
                            in_str.insert(i + 1, TokenDicRev[TokenType.R_PARANTHESIS])
                    i += 1
            i += 1

        for i, token in enumerate(in_str):
            if token in TokenDic.keys():
                error_conditions = [
                    token in [TokenDicRev[TokenType.LOGIC_NOT], TokenDicRev[TokenType.L_PARANTHESIS]]\
                        and (tokens and tokens[-1].type in [TokenType.SYMBOL, TokenType.R_PARANTHESIS]),
                    token in [TokenDicRev[TokenType.R_PARANTHESIS]] and tokens[-1].type == TokenType.L_PARANTHESIS,
                    token in [TokenDicRev[TokenType.LOGIC_OR], TokenDicRev[TokenType.LOGIC_AND]]\
                        and i == len(in_str) - 1 or (tokens and tokens[-1] == TokenDicRev[TokenType.L_PARANTHESIS])
                ]
                if True in error_conditions:
                    raise Exception(f'Invalid syntax at {token} ({i})')
                tokens.append(ExpressionToken(TokenDic[token], 0))
            elif re.match(r"[a-zA-Z]+\d*", token):
                tokens.append(ExpressionToken(TokenType.SYMBOL, token))
            else:
                raise Exception(f'Illegal token {token}')

        token_cache[id] = tokens
        return tokens

    def _convert_to_postfix(self, tokens):
        """Convert tokenized expression to postfix form"""
        stack = deque()
        output = []
        for token in tokens:
            if token.type == TokenType.LOGIC_AND or token.type == TokenType.LOGIC_OR:
                if stack:
                    minPrecedence = Precedence[token.type]
                    while Precedence[stack[-1].type] >= minPrecedence:
                        output.append(stack.pop())
                        if not stack:
                            break
                stack.append(token)
            elif token.type == TokenType.SYMBOL:
                output.append(token)
            elif token.type in [TokenType.LOGIC_NOT, TokenType.L_PARANTHESIS, TokenType.LOGIC_HIGH, TokenType.LOGIC_LOW]:
                stack.append(token)
            elif token.type == TokenType.R_PARANTHESIS:
                if not stack:
                    raise Exception("Missing parenthesis")
                while stack[-1].type != TokenType.L_PARANTHESIS:
                    output.append(stack.pop())
                    if not stack:
                        raise Exception("Missing parenthesis")
                stack.pop()
        if not stack:
            return output
        if stack[-1].type == TokenType.L_PARANTHESIS:
            raise Exception("Missing parenthesis")
        while stack:
            output.append(stack.pop())
        return output

    def _evaluate_symbol(self, symbol):
        """Evaluate the symbol"""
        if symbol not in self.dictionary:
            raise Exception(f'Symbol {symbol} doesn\'t exist.')
        return self.dictionary[symbol]

    @classmethod
    def get_symbols(cls, id, in_str):
        tokens = cls._tokenize(id, in_str)
        return [token.value for token in tokens if token.type == TokenType.SYMBOL and token.value not in [TokenDicRev[TokenType.LOGIC_HIGH], TokenDicRev[TokenType.LOGIC_LOW]]]

    def evaluate(self, id, in_str):
        """Evaluate string expression"""
        if not in_str:
            raise Exception('Empty expression string')
        tokens = self._convert_to_postfix(self._tokenize(id, in_str))
        stack = deque()
        result = False
        for token in tokens:
            if token.type == TokenType.LOGIC_NOT:
                rightOperand = stack.pop()
                result = not rightOperand
                stack.append(result)
            elif token.type in [TokenType.LOGIC_AND, TokenType.LOGIC_OR]:
                rightOperand = stack.pop()
                leftOperand = stack.pop()
                result = leftOperand and rightOperand if token.type == TokenType.LOGIC_AND else leftOperand or rightOperand
                stack.append(result)
            elif token.type == TokenType.SYMBOL:
                stack.append(self._evaluate_symbol(token.value))
            elif token.type == TokenType.LOGIC_HIGH:
                stack.append(True)
            elif token.type == TokenType.LOGIC_LOW:
                stack.append(False)
        return stack.pop()

# Test
if __name__ == "__main__":
    evaluator = LogicExpressionEvaluator({
        "u": False,
        "x1": False,
        "x2": False,
        "x3": True,
        "x4": False
    })
    expression = "not x4 and not u and (x2 or x3)"
    print(evaluator.evaluate(expression))
