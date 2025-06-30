import pyparsing as pp

# --- Define basic lexical elements ---
IDENTIFIER = pp.Word(pp.alphas, pp.alphanums + "_")

# Keywords as pyparsing Literal objects, then suppressed
KW_DEFINE = pp.Suppress(pp.Keyword("Define"))
KW_AS_A = pp.Suppress(pp.Keyword("as a"))
KW_HAS = pp.Suppress(pp.Keyword("has"))
KW_WITH = pp.Suppress(pp.Keyword("with"))
KW_IS = pp.Suppress(pp.Keyword("is"))
KW_A_SUB_CLASS_OF = pp.Suppress(pp.Literal("a sub-class of"))  # Literal for multi-word
DOT = pp.Suppress(pp.Literal("."))

# Define VALUE carefully. Order matters here too!
QUOTED_STRING = pp.quoted_string
NUMBER = pp.Word(pp.nums)

# --- Define the DSL grammar rules ---

# Rule 4: Class is a sub-class of SuperClass.
subclass_stmt = (
        IDENTIFIER("class_name") +
        KW_IS +
        KW_A_SUB_CLASS_OF +
        IDENTIFIER("super_class_name") +
        DOT
).set_results_name("subclass")

# Rule 1a: Define Object as a Class.
define_object_as_class_stmt = (
        KW_DEFINE +
        IDENTIFIER("object_name") +
        KW_AS_A +
        IDENTIFIER("class_name") +
        DOT
).set_results_name("define_as_class")

# Rule 1b: Define Object. (for existence)
define_object_only_stmt = (
        KW_DEFINE +
        IDENTIFIER("object_name") +
        DOT
).set_results_name("define_only")

# Rule 2a: Object has Property with Value.
object_has_property_with_value_stmt = (
        IDENTIFIER("object_name") +
        KW_HAS +
        IDENTIFIER("property_name") +
        KW_WITH +
        (QUOTED_STRING("value") | NUMBER("value") | IDENTIFIER("value")) +
        DOT
).set_results_name("has_with_value")

# Rule 2b: Object has Property. (implicitly any value)
object_has_property_only_stmt = (
        IDENTIFIER("object_name") +
        KW_HAS +
        IDENTIFIER("property_name") +
        DOT
).set_results_name("has_only")

# Rule 3: Object is Property. (Boolean property)
object_is_boolean_stmt = (
        IDENTIFIER("object_name") +
        KW_IS +
        IDENTIFIER("boolean_property_name") +
        DOT
).set_results_name("is_boolean")

# Combine all statement types into a single parser
# Critical: Order rules from most specific to least specific.
dsl_parser = (
        subclass_stmt |  # e.g., Student is a sub-class of Person.
        define_object_as_class_stmt |  # e.g., Define John as a Person.
        object_has_property_with_value_stmt |  # e.g., John has age with 30.
        object_has_property_only_stmt |  # e.g., John has a car.
        object_is_boolean_stmt |  # e.g., Mary is vegetarian.
        define_object_only_stmt  # e.g., Define Mary.
)


# --- Preprocessor Logic ---
def preprocess_dsl_statement_pyparsing(dsl_statement: str) -> str:
    try:
        # Parse the statement
        parsed_results = dsl_parser.parse_string(dsl_statement.strip(), parseAll=True)

        # Get the results name to determine which rule matched
        results_name = parsed_results.get_name()

        if results_name == "define_as_class":
            obj = parsed_results.object_name
            cls = parsed_results.class_name
            return f"{obj}:{cls}."
        elif results_name == "define_only":
            obj = parsed_results.object_name
            return f"{obj}[]."
        elif results_name == "has_with_value":
            obj = parsed_results.object_name
            prop = parsed_results.property_name
            value = parsed_results.value

            if isinstance(value, str):
                if value.startswith("'") or value.startswith('"'):
                    return f"{obj}[{prop} -> {value}]."  # Value already has quotes
                elif value.isdigit():
                    return f"{obj}[{prop} -> {value}]."
                else:  # Otherwise, assume it's an F-logic symbol that needs quoting
                    return f"{obj}[{prop} -> '{value}']."
            else:  # If pyparsing converted it to int/float directly
                return f"{obj}[{prop} -> {value}]."

        elif results_name == "has_only":
            obj = parsed_results.object_name
            prop = parsed_results.property_name
            return f"{obj}[{prop} -> ?_any]."
        elif results_name == "is_boolean":
            obj = parsed_results.object_name
            prop = parsed_results.boolean_property_name
            return f"{obj}[{prop}]."
        elif results_name == "subclass":
            sub_cls = parsed_results.class_name
            super_cls = parsed_results.super_class_name
            return f"{sub_cls}::{super_cls}."

    except pp.ParseException as pe:
        return f"// ERROR (Parsing): {pe}\n// Original: {dsl_statement}"
    except Exception as e:
        return f"// ERROR (Unexpected): {e.__class__.__name__}: {e}\n// Original: {dsl_statement}"

    return f"// ERROR (Unhandled): {dsl_statement}"


# --- Test cases ---
dsl_statements = [
    "Define John as a Person.",
    "Define Mary.",
    "John has age with 30.",
    "Mary has favorite_color with 'blue'.",
    "Company has name with \"Acme Corp\".",  # Test double quotes
    "John has a car.",  # Implicit value test
    "Mary is vegetarian.",
    "Student is a sub-class of Person.",
    "MyObject has status with Active.",  # Test unquoted string value
    "Define 123.",  # Test numeric object name
    "Employee is a sub-class of Worker.",
    "Invalid statement here."  # Test error handling with a dot
]

print("--- DSL Input ---")
for s in dsl_statements:
    print(s)

print("\n--- F-logic Output ---")
for s in dsl_statements:
    print(preprocess_dsl_statement_pyparsing(s))