###############################################################################
import re
import random

class VariableStructure:
    """
    -Contains null elements but no remaining ones
    -Will contain fragmented iterables if elements are added
    -An initialised list will have elements stored sequentially
    -A list reference takes up one element and stores points to first element
    -When deleting a list, it will sequentially step through each element by jumping to pointer locations
    -Each element contains a value and a pointer to the next value
    """
    def __init__(self):
        self.memory = [] # List of single objects
        self.call_stack = []
    
    def __repr__(self):
        return "\n".join([str(i) + "," + str(self.memory[i]) for i in range(len(self.memory))])
    
    def __iter__(self):
        for el in self.memory:
            yield el
    
    def __getitem__(self, index):
        return self.memory[index]

    def find_next_space(self):
        for i in range(len(self.memory)):
            if self.memory[i].is_null:
                return i

    def pointer_of(self, index):
        return self.memory[index].pointer

    def _append(self, to_add):
        location = self.find_next_space()
        if location is None:
            self.memory.append(to_add)
            return len(self.memory) - 1
        else:
            self.memory[location] = to_add
            return location
    
    def append(self, value=None, next=None):
        return self._append(VariableStructureElement(value, next))

    def append_ptr(self, to=None, next=None):
        return self._append(IterableStartElement(to, next))

    def append_general(self, value=None):
        if type(value) == list:
            return self.append_list(value)
        else:
            return self.append(value)

    def replace(self, index, obj):
        self.memory[index] = obj
    
    def get_element(self, start, pos):
        indexes = tuple(self.element_indexes(start))
        try:
            return indexes[pos]
        except IndexError:
                raise IndexError(f"Index {pos} is too high")

    def delete_general(self, index):
        if type(self[index]) == IterableStartElement:
            self.delete_list(index)
        else:
            self.clear_cell(index)

    def clear_cell(self, index):
        self.replace(index, VariableStructureElement())
        while self[-1].value == None:
            del self.memory[-1]

    def delete_list(self, start, pos=None): # MEMORY LEAK
        #print(f"start: {start}, pos: {pos}")
        if pos is None:
            for index in tuple(self.element_indexes(start)):
                #print("Deleting", index, self[index])
                if type(self[index]) == IterableStartElement:
                    self.delete_list(index)
                else:
                    self.clear_cell(index)
            self.clear_cell(start)
            return start
        else:
            indexes = tuple(self.element_indexes(start))
            i = indexes[pos]
            if pos == 0: # deleted first element
                self.clear_cell(indexes[0])
                if len(indexes) == 1:
                    self[start].start_location = None
                else:
                    self[start].start_location = indexes[1]
            else:
                if pos + 1 == len(indexes): # deleted last element
                    self[indexes[pos - 1]].pointer = None
                else:
                    self[indexes[pos - 1]].pointer = indexes[pos + 1]
            if type(self[i]) == IterableStartElement:
                self.delete_list(i) # Delete the lis that starts at the location of the element to be deleted
            else:
                self.clear_cell(i)
            return  i # the location of the deleted element
    
    def replace_element(self, start, index, new):
        self.delete_list(start, index)
        self.extend_insert(start, index, new)
        
    def element_indexes(self, start):
        current = self[start].start_location
        while True:
            yield current
            current = self[current].pointer
            if current is None:
                break

    def append_list(self, obj): 
        prev_location = list_location = self.append_ptr()
        for i in range(len(obj)):
            # Add new object with no next value
            # Set next of previous to current
            #print(i, prev_location, self[-1])
            new_location = self.append_general(obj[i])
            #print(prev_location)
            setattr(self.memory[prev_location], "start_location" if i == 0 else "pointer", new_location)
            prev_location = new_location
        return list_location
    
    def find_end(self, start):
        current = self[start].start_location
        while True:
            next = self[current].pointer
            if next is None:
                return current
            current = next

    def obtain(self, index):
        """ If it's an iterable start, look at pointer and go from there
        Otherwise, go to next
        """
        pointers = []
        compiled = None
        while True:
            current_element = self.memory[index]
            #print(f"`{index}`, `{compiled}`, `{current_element}`, `{pointers}`")
            if type(current_element) == IterableStartElement:
                pointers.append(index) # so it knows where to go when it reaches the end
                if compiled is None:
                    compiled = []
                else:
                    compiled.append([]) # instantiate with no arguments
                index = current_element.start_location
            else:
                if compiled is None:
                    return current_element.value
                append_to = compiled
                for _ in range(len(pointers) - 1):
                    append_to = append_to[-1]
                append_to.append(current_element.value)
                index = current_element.pointer
            while index is None:
                if len(pointers) == 0:
                    return compiled
                index = self.memory[pointers.pop()].pointer
            #print("Going to", index)
    
    def extend_end(self, start, value):
        end = self.find_end(start)
        self[end].pointer = self.append_general(value)
    
    def extend_insert(self, start, pos, value):
        # If pos = 0, modify start of list
        # otherwise modify pointer of previous
        new_location = self.append_general(value)
        if pos == 0:
            if self[start].start_location is not None:
                self[new_location].pointer = self[start].start_location
            self[start].start_location = new_location
        else:
            current = self[start].start_location
            for _ in range(pos - 1):
                current = self[current].pointer
            next = self[current].pointer
            #print(current, self[current])
            #print(next, self[next])
            new_location = self.append_general(value)
            self[current].pointer = new_location
            self[new_location].pointer = next

    
    """
    def _obtain(self, index):
        current_element = self.memory[index]
        if type(current_element) == IterableStartElement:
            self.obtain(current_element.start_location)
        else:
    """ 
class VariableStructureElement:
    def __init__(self, value=None, pointer=None):
        self.value = value
        self.pointer = pointer
    
    def __repr__(self):
        return f"{self.value},->{self.pointer}"
    
    @property
    #@functools.lru_cache()
    def is_null(self):
        return self.value is self.pointer is None

class IterableStartElement:
    def __init__(self, start_location, pointer):
        self.start_location = start_location
        self.pointer = pointer
        self.is_null = False
    
    def __str__(self):
        return f"List@{self.start_location},->{self.pointer}"

letters = "abcdefghijklmnopqrstuvwxyz"
letters += letters.upper()

function = type(lambda: None)
get_last = lambda lst: lst[-1]

def getter_function(x, y):
    if len(y) != 1:
        raise TypeError("Must use one argument when getting list item")
    if y[0] % 1 != 0:
        raise TypeError("Cannot get list item using float")
    try:
        return x[int(y[0])]
    except TypeError as e:
        #print(e)
        raise TypeError(f"Must get list item using integers;'{y[0]}' is invalid")

op_functions = {
    "POW": lambda x, y: x ** y,
    "USUB": lambda x: -x,
    "MUL": lambda x, y: x * y,
    "DIV": lambda x, y: x / y,
    "FDIV": lambda x, y: x // y,
    "SUB": lambda x, y: x - y,
    "ADD": lambda x, y: x + y,
    "NE": lambda x, y: x != y,
    "EQ": lambda x, y: x == y,
    "LT": lambda x, y: x < y,
    "GT": lambda x, y: x > y,
    "GE": lambda x, y: x >= y,
    "LE": lambda x, y: x <= y,
    "SET": lambda x, y: x.assign(y), # where x is a reference
    "GET": getter_function
}

op_text = {
    "POW": lambda x, y: f"{x} to the power of {y}",
    "USUB": lambda x: f"{x} but negated",
    "MUL": lambda x, y: f"{x} multiplied by {y}",
    "DIV": lambda x, y: f"{x} divided by {y}",
    "FDIV": lambda x, y: f"floor division of {x} and {y}",
    "SUB": lambda x, y: f"{x} minus {y}",
    "ADD": lambda x, y: f"{x} plus {y}",
    "SET": lambda x, y: f"set {x} to {y}",
    "GET": lambda x, y: f"element {y} of {x}",
    "LT": lambda x, y: f"{x} is less than {y}",
    "GT": lambda x, y: f"{x} is greater than {y}",
    "LE": lambda x, y: f"{x} is less than or equal to {y}",
    "GE": lambda x, y: f"{x} is greated than or equal to {y}",
    "EQ": lambda x, y: f"{x} equals {y}"
    
}

unary = {"USUB"}

def print_(*args):
    print("OUTPUT: ", end="")
    for x in args:
        if type(x) == Reference:
            x.location_check()
        print(x, end="")
    print()

def find_name(token, is_unary):
    for level in precedence:
            for op, sym in level.items():
                if ((is_unary and op in unary) or (not is_unary and op not in unary)) and sym == token:
                    return op
    raise SyntaxError(f"'{token}' is not a valid  operator")

precedence = (
    {},
    {"POW": "^"},
    {"USUB": "-"},
    {"MUL": "*", "DIV": "/", "FDIV": "//"},
    {"ADD": "+", "SUB": "-"},
    {"NE": "!=", "EQ": "=="},
    {"LT": "<", "GT": ">", "LE": "<=", "GE": ">="},
    {"SET": "="}
)

flat_operators = set()
for ops in precedence:
    for sym in ops.values():
        flat_operators.add(sym)

right_associative = {"POW", "SET"}

def get_prec(token):
    if token in {"OPENPAR", "OPENSQ", "OPENCALL", "GET"}:
        return 0
    for level in range(len(precedence)):
        if token in precedence[level].keys():
            return level
    raise IndexError(token)

def amount_to_pop(operator, stack):
    op_prec = get_prec(operator)
    index = len(stack) - 1
    while index >= 0:
        if get_prec(stack[index]) >=  op_prec or stack[index] == "OPENSQ":
            break
        index -= 1
    return len(stack) - 1 - index

def is_operator(string):
    try:
        return string in flat_operators
    except KeyError:
        return False

def op_function(string):
    if is_operator(string):
        for prec in precedence:
            for k, v in prec.items():
                if v == string:
                    return k
    raise SyntaxError("Unrecognised operator")

def needs_def(func):
    def wrapped(self, other):
        #print(self.location, other)
        #self.location_check()
        if type(other) == Reference:
            other_ = Program.memory.obtain(other.location)
        else:
            other_ = other
        return func(Program.memory.obtain(self.location), other_)
    return wrapped


class Reference:
    def __init__(self, name, location=None, index=None):
        #print("New ref at", location)
        self.location = location
        self.name = name
        self.index = index
    
    def __repr__(self):
        if self.location is None:
            return "<Undefined>"
        if self.index is None: # single variable
            return str(Program.memory.obtain(self.location))
        return str(Program.memory[Program.memory.get_element(self.location, self.index)].value)

    def display_loc(self):
        if self.index is None:
            return self.name
        return self.name + "[" + str(self.index) + "]"

    def assign(self, other):
        #print("Assignment", self.location, self.index)
        if type(other) == Reference:
            other.location_check()
        if self.location is None: # then so is index
            self.location = Program.memory.append_general(other)
            Program.scope_stack[-1][self.name] = self.location
        elif self.index is None:
            Program.memory[self.location].value = other
        else:
            Program.memory.replace_element(self.location, self.index, other)

    def location_check(self):
        if self.location is None:
            raise NameError("Trying to operate on uninitialised variable")
    
    def __call__(self, *args):
        #print("CALLING", args)
        #print(self.location)
        return Program.memory.obtain(self.location)(*args)

    @needs_def
    def __add__(self, other):
        return self + other
    
    @needs_def
    def __sub__(self, other):
        return self - other
    
    @needs_def
    def __truediv__(self, other):
        return self / other
    
    @needs_def
    def __rtruediv__(self, other):
        return other / self
    
    @needs_def
    def __floordiv__(self, other):
        return self // other
    
    @needs_def
    def __rfloordiv__(self, other):
        return other // self
    
    @needs_def
    def __mul__(self, other):
        return self * other
    
    @needs_def
    def __rmul__(self, other):
        return other * self
    
    @needs_def
    def __pow__(self, other):
        return self ** other

    @needs_def
    def __rpow__(self, other):
        return self ** other
    
    @needs_def
    def __lt__(self, other):
        return self < other
    
    @needs_def
    def __gt__(self, other):
        return self > other
    
    @needs_def
    def __eq__(self, other):
        return self == other
    
    #@needs_def
    def __getitem__(self, other):
        #print(Program.memory.get_element(self.location, other))
        return Reference(None, self.location, other)

class ReturnEvent:
    def __init__(self, value):
        self.value = value

class Singleton:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

NULL = Singleton("NULL")

class Function:
    def __init__(self, name, start, end, args):
        self.name = name
        self.args = tuple(args)
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.name}({','.join(self.args)})"
    
    def __call__(self, *args):
        print(f"We are now in the function {self.name} with {list_to_text(args)} passed in.")
        if self.start == self.end: # no code
            return
        #Program.scope_stack.append()
        new = {}
        for i in range(len(args)):
            if type(args[i]) == Reference:
                # should add a new variable to scope which references same
                if args[i].index is None:
                    new[self.args[i]] = args[i].location
                else:
                    new[self.args[i]] = Program.memory.append_general(Program.memory.get_element(args[i].location, args[i].index))
            else:
                new[self.args[i]] = Program.memory.append_general(args[i])
        Program.scope_stack.append(new)
        return_line = Program.line_n
        Program.line_n = self.start
        while Program.line_n != self.end:
            result = Program.run_single_line(Program.modified_code[Program.line_n])
            if type(result) == ReturnEvent:
                #print(result.value, type(result.value))#, Program.memory.obtain(result.value.location))
                #input("...")
                break
            Program.line_n += 1
        Program.line_n = return_line
        #print("returning to line", return_line + 1)
        if type(result.value) == Reference:
            return_value = Program.memory.obtain(result.value.location)
        else:
            return_value = result.value
        for v in Program.scope_stack[-1].values():
            if not Program.contains_reference(v):
                Program.memory.delete_general(v)
        del Program.scope_stack[-1]
        return return_value


class String(str):
    def __repr__(self):
        return f"String({self})"

def list_to_text(array):
    if len(array) == 0:
        return "no arguments"
    array = tuple(map(str, array))
    if len(array) == 1:
        return "one argument: " + array[0]
    return "arguments [" + ", ".join(array[:-1]) + " and " + array[-1] + "]"

def find_last_line(indents, start):
    lines = start + 1
    while lines < len(indents) and indents[start] != indents[lines]:
            lines += 1
    return lines

class Program:
    @classmethod
    def contains_reference(cls, value):
        for scope in cls.scope_stack[:-1]:
                if value in scope.values():
                    return True
        return False
    @staticmethod
    def tokenise(expression):
        #raise Exception("Don't forget about strings!")
        prev = ""
        current = ""
        prev_type = None
        in_string = False
        tokens = []
        for char in expression:
            if in_string:
                if char == "\"":
                    in_string = False
                    tokens.append(String(current))
                    current = ""
                else:
                    current += char
                continue
            found_op = False
            #print(char, end=" ")
            if prev_type == "operator":
                for operator in flat_operators:
                    if current + char == operator[:len(current) + 1]:
                        # possible valid operator
                        #print(current + char, "is possible")
                        current += char
                        prev_type = "operator"
                        found_op = True
            if not found_op:
                if char == "\"": # not in string - start
                    if current != "":
                        tokens.append(current)
                    in_string = True
                    current = ""
                if char.isdigit():
                    # A digit can appear after a variable, a number or a dot but not after a closing bracket
                    if prev_type == "close":
                        raise SyntaxError("Number after closing bracket")
                    elif prev_type == "operator":
                        tokens.append(current)
                        current = char
                    else:
                        current += char
                    prev_type = "digit"
                elif char in letters:
                    if prev_type == "digit":
                        raise SyntaxError("Letter after number")
                    elif prev_type == "letter":
                        # append to variable name
                        current += char
                    else:
                        if current != "":
                            tokens.append(current)
                        current = char
                    prev_type = "letter"
                elif char == ".":
                    if "." in current:
                        raise SyntaxError("Out of place dot")
                    if current[0] in letters: # after a variable
                        raise SyntaxError("No OOP here I'm afraid")
                    if prev_type == "digit":
                        current += "."
                    prev_type = "dot"
                elif char in "([])":
                    if current != "":
                        tokens.append(current)
                    tokens.append(char)
                    current = ""
                    if char in "([":
                        prev_type = "open"
                    else:
                        prev_type = "close"
                elif char in {string[0] for string in flat_operators}:
                    if current != "":
                        tokens.append(current)
                    current = char
                    prev_type = "operator"
                elif char == ",":
                    if current != "": 
                        tokens.append(current)
                    tokens.append(",")
                    current = ""
                    prev_type = "comma"
            #print(prev_type, repr(current), tokens)
            #prev = char
        if current != "":
            tokens.append(current)
        return tokens

    @staticmethod
    def token_types(tokens):
        types = []
        for token in tokens:
            if type(token) == String:
                types.append("string")
            elif token == "(":
                types.append("openpar")
            elif token == ")":
                types.append("closepar")
            elif token == "[":
                types.append("opensq")
            elif token == "]":
                types.append("closesq")
            elif token == ",":
                types.append("comma")
            elif token in flat_operators:
                types.append("operator")
            elif re.match(r"\d+(\.\d+)?", token):
                types.append("number")
            elif re.match(r"[a-zA-Z]([a-zA-Z0-9])*", token):
                types.append("identifier")
            else:
                raise Exception(f"What is the data type for '{token}'?")
        return types
    
    @staticmethod
    def create_postfix(tokens, types):
        operators = [] # Cyclomatic complexity is 27, apparently
        postfix = []
        for i in range(len(tokens)):
            if types[i] == "operator":
                if i == 0 or types[i - 1] in {"operator", "opensq", "openpar"}:
                        # it's unary
                        tokens[i] = find_name(tokens[i], is_unary=True)
                else:
                    tokens[i] = find_name(tokens[i], is_unary=False)
        for i in range(len(tokens)):
            t_type = types[i]
            token = tokens[i]
            #print(token, t_type)
            if t_type == "number":
                postfix.append(float(token))
            elif t_type == "string":
                postfix.append(token)
            elif t_type == "identifier":
                if token in {"OPENCALL", "CLOSECALL"}:
                    raise NameError(f"Name '{token}' is reserved")
                postfix.append(token)
            elif t_type == "operator":
                if i != 0 and len(operators) != 0 and operators[-1] not in {"OPENPAR", "OPENSQ", "OPENCALL"}:
                    token_prec = get_prec(token)
                    stack_prec = get_prec(operators[-1])
                    #print("token", token, stack_prec, token_prec)
                    #print(token_prec, stack_prec, amount_to_pop(token, operators))
                    if token_prec == stack_prec:
                        if not (token in right_associative or token in unary):
                            postfix.append("__" + operators.pop())
                    else:
                        pop_amount = amount_to_pop(token, operators)
                        #print("pop", pop_amount)
                        for _ in range(pop_amount):
                            postfix.append("__" + operators.pop())
                operators.append(token)
            elif t_type == "openpar":
                if i != 0 and types[i - 1] != "operator": # call to fn
                    postfix.append("OPENCALL")
                    operators.append("OPENCALL")
                else: # to change order
                    operators.append("OPENPAR")
            elif t_type == "closepar":
                while operators[-1] not in {"OPENPAR", "OPENCALL"}:
                    postfix.append("__" + operators.pop())
                if operators.pop() == "OPENCALL":
                    postfix.append("CLOSECALL")
            elif t_type == "comma":
                while operators[-1] not in {"OPENPAR", "OPENCALL", "OPENSQ"}:
                    postfix.append("__"+ operators.pop())
            elif t_type == "opensq":
                if i != 0 and types[i - 1] not in {"operator", "comma", "opensq", "openpar"}: # get item from list
                    operators.append("GET")
                operators.append("OPENSQ")
                postfix.append("OPENSQ")
            elif t_type == "closesq":
                while True:
                    if operators[-1] == "OPENSQ":
                        break
                    postfix.append("__" + operators.pop())
                operators.pop()
                postfix.append("CLOSESQ")
            #print(postfix, operators)
        while operators:
            postfix.append("__" + operators.pop())
        return postfix

    @classmethod
    def eval_postfix(cls, postfix):
        stack = []
        text = []
        for token in postfix:
            #print(token)
            if type(token) == float:
                stack.append(token)
                text.append(str(token))
            elif type(token) == String:
                stack.append(token)
                text.append(f'the string \"{token}\"')
            elif token == "OPENSQ":
                stack.append("[")
            elif token == "CLOSESQ":
                new_list = []
                new_text_list = []
                while stack[-1] != "[":
                    new_list.insert(0, stack.pop())
                    new_text_list.insert(0, text.pop())
                del stack[-1]
                stack.append(new_list)
                text.append("list [" + ", ".join(new_text_list) + "]")
            elif token == "OPENCALL":
                stack.append(token)
                text[-1] = "the function " + text[-1] + " called with "
            elif token == "CLOSECALL":
                args = []
                text_args = []
                while stack[-1] != "OPENCALL":
                    args.insert(0, stack.pop())
                    text_args.insert(0, text.pop())
                stack.pop() # remove OPENCALL
                text[-1] += list_to_text(text_args)
                stack[-1] = stack[-1](*args)
            elif re.match(r"__[A-Z]+", token):
                operator = token[2:]
                if operator in op_functions:
                    if operator in unary:
                        #stack[-1] = op_functions[operator](stack[-1])
                        arguments = (stack[-1],)
                        del stack[-1]
                    else:
                        top = (stack.pop(), stack.pop())
                        top_text = (text.pop(), text.pop())
                        #stack.append(op_functions[operator](top[1], top[0]))
                        arguments = (top[1], top[0])
                        text_arguments = (top_text[1], top_text[0])
                    stack.append(op_functions[operator](*arguments))
                    text.append(op_text[operator](*text_arguments))
                else:
                    raise NameError("Double underscore reserved")
            elif re.match(r"[a-zA-Z][a-zA-Z0-9]*", token): # variable
                #print("REFERENCE:", token)
                #print(cls.scope_stack[-1].keys())
                for scope in cls.scope_stack[::-1]:
                    if token in scope.keys():
                        stack.append(Reference(token, scope[token]))
                        break
                else:
                    stack.append(Reference(token))
                text.append(stack[-1].display_loc())
                """
                try:
                    
                    stack.append(scope[token])
                except KeyError:
                    raise NameError(f"'{token}' is not defined")"""
            #print("stack:", stack)
            #print("text:", text)
        if len(stack) != 1:
            raise SyntaxError("Operands do not match operators")
        return stack[0], " ".join(text)

    @classmethod
    def text_to_postfix(cls, text):
        tokens = cls.tokenise(text)
        #print(tokens)
        types = cls.token_types(tokens)
        postfix = cls.create_postfix(tokens, types)
        return postfix

    @classmethod
    def construct_code(cls, text):
        modified = []
        lines = [el for el in text.split("\n") if el != ""]
        print(lines)
        indents = [(len(line) - len(line.lstrip())) // 4 for line in lines] + [0]
        print(indents)
        n = 0
        last_if = {}
        while n < len(lines):
            line = lines[n].lstrip()
            if "#" in line:
                line = line[:line.index("#")]
            if re.match(r"^(\s*)$", line): # Ignore line
                n += 1
                continue
            #print("READING: '{}' @ {}".format(line, indents[n]))
            i = 0
            while i < len(line) and line[i] != " ":
                i += 1
            command = line[:i]
            #print("command:", repr(command))
            if command == "FUNC":
                try:
                    j = i
                    while line[j] != "(":
                        j += 1
                    f_name = line[i+1:j] # name of function
                    remainder = line[j+1:-1].replace(" ", "")
                    args = () if remainder == "" else remainder.split(",")
                    """
                    in_fn = n + 1
                    while in_fn < len(lines) and indents[n] != indents[in_fn]:
                        in_fn += 1
                    """
                    modified.append(("FUNC", f_name, n + 1, find_last_line(indents, n), tuple(args)))
                except IndexError:
                    raise SyntaxError("Invalid function definition")
            elif command == "RETURN":
                modified.append(("RETURN", cls.text_to_postfix(line[i:])))
            elif command == "WHILE":
                modified.append(("WHILE", cls.text_to_postfix(line[i:]), find_last_line(indents, n))) # condition, ending line
            elif command == "IF":
                last_if[indents[n]] = n
                modified.append(("IF", find_last_line(indents, n) - 1, cls.text_to_postfix(line[i:])))
            elif command == "ELIF":
                if indents[n] in last_if.keys():
                    last_if[indents[n]] = n
                else:
                    raise SyntaxError("ELIF with no preceding IF")
                modified.append(("ELIF", find_last_line(indents, n) - 1, cls.text_to_postfix(line[i:])))
            elif command == "ELSE":
                if indents[n] in last_if.keys():
                    last_if[indents[n]] = n
                    modified.append(("ELSE", find_last_line(indents, n) - 1))
                    #modified[last_if[indents[n]]] += n
                    del last_if[indents[n]]
                else:
                    raise SyntaxError("ELSE with no preceding IF or ELIF")
            else:
                modified.append(("EXP", cls.text_to_postfix(line)))
            n += 1
        #print(*modified, sep="\n")
        cls.modified_code = modified + ["END"]#[("EXP", ["print", "OPENCALL", String("The program exited successfully"), "CLOSECALL"])]
        print(*cls.modified_code, sep="\n")
    
    @classmethod
    def run_modified(cls, **mapping):
        try:
            cls.scope_stack.append({})
            for k, v in mapping.items():
                Program.scope_stack[-1][k] = Program.memory.append_general(v)
            cls.line_n = 0
            while cls.line_n < len(cls.modified_code):
                cls.run_single_line(cls.modified_code[cls.line_n])
                cls.line_n += 1
        except Exception as e:
            print("We encountered an exception:", e)
        else:
            print("We have successfully exited the program. *shines*")
    
    @classmethod
    def run_single_line(cls, line):
        #print("EXECUTING:", repr(line))
        #print("Memory before execution:", cls.memory)
        if line == "END":
            print("Goodbye!")
        if line[0] == "EXP":
                result, text = cls.eval_postfix(line[1])
                print("We just evaluated the expression: ", text)
        elif line[0] == "FUNC":
            fn_text = f"We will define the function '{line[1]}' which takes "
            fn_text += list_to_text(line[4])
            print(fn_text)
            cls.scope_stack[-1][line[1]] = cls.memory.append(Function(*line[1:]))
            #print("Going to", cls.modified_code[line[3]])
            cls.line_n = line[3] - 1
        elif line[0] == "WHILE":
            result, text = cls.eval_postfix(line[1])
            print(f"Is '{text}' true? *pause*")
            if result:
                print("\tYes, so we execute the code in the while loop")
                while_loop_line = cls.line_n
                cls.line_n += 1
                while cls.line_n < line[2]: # while line is in while loop
                    cls.run_single_line(cls.modified_code[cls.line_n])
                    cls.line_n += 1
                cls.line_n = while_loop_line - 1
                print()
            else:
                print("\tNo, so we exit the while loop")
                cls.line_n = line[2] - 1
        elif line[0] in {"IF", "ELIF"}:
            result, text = cls.eval_postfix(line[2])
            print(f"Is '{text}' true? *pause*")
            if result:
                print("\tYes, so we execute the code below")
                while cls.line_n < line[1] - 1:
                    cls.line_n += 1
                    #print("Line:", cls.line_n)
                    cls.run_single_line(cls.modified_code[cls.line_n])
                cls.line_n = line[1]
                while cls.modified_code[cls.line_n][0] in {"ELIF", "ELSE"}:
                    cls.line_n = cls.modified_code[cls.line_n][1]
                #print("Finally,", cls.line_n)
            else:
                print("\tNo", end="")
                if cls.modified_code[line[1]][0] in {"ELIF", "ELSE"}:
                    print(", so we move onto the following " + cls.modified_code[line[1]][0])
                else:
                    print()
                cls.line_n = line[1] - 1
        elif line[0] == "ELSE":
            print("We are now going to execute the code below the ELSE")
            while cls.line_n < line[1]:
                    cls.line_n += 1
                    #print("Line:", cls.line_n)
                    cls.run_single_line(cls.modified_code[cls.line_n])
        elif line[0] == "RETURN":
            result, text = cls.eval_postfix(line[1])
            print("From the function, we return:", text)
            return ReturnEvent(result)
    
    memory = VariableStructure()
    scope_stack = []
    text_lines = []

"""
func foo(x, y)
	print x + y
loop i 0->5
	foo(i, i^2)
j = 5
while j < 5
	j = j + 1
"""
Program.construct_code(open("code.txt", "r").read())
Program.run_modified(print=print_, rand=random.randint, **op_functions)
print("\n\n")
"""
memory = VariableStructure()
memory.appenProgram.construct_code(open("code.txt", "r").read())d_list(["a", "b", "c", "d", ["e", "f", "g"], "h", "i"])
print(memorymemory.delete_list(0, 1)
print()
print(memory)
print(memory.obtain(0))
""
print(memory, "\n")
#print(memory.find_end(0))
memory.extend_insert(0, 0, "!")
print(memory.obtain(0))
index = memory.element_indexes(0)
while True:
    print(next(index))
"""