import operator
import numbers
from z3 import *


class Z3I:
    def __init__(self, z3_var):
        self.z3_var = z3_var

    @staticmethod
    def _unwrap(other):
        if isinstance(other, Z3I):
            return other.z3_var
        elif isinstance(other, numbers.Number):
            return other
        elif isinstance(other, bool):
            return other
        else:
            raise Exception(f"Cannot unwrap {other}")

    def __add__(self, other):
        return Z3I(operator.add(self.z3_var, self._unwrap(other)))

    # Similarly, other arithmetic operations
    def __sub__(self, other):
        return Z3I(operator.sub(self.z3_var, self._unwrap(other)))

    def __mul__(self, other):
        return Z3I(operator.mul(self.z3_var, self._unwrap(other)))

    # ...
    # what if a number is on the left side of the operator?
    def __radd__(self, other):
        return Z3I(operator.add(self._unwrap(other), self.z3_var))

    def __rsub__(self, other):
        return Z3I(operator.sub(self._unwrap(other), self.z3_var))

    def __rmul__(self, other):
        return Z3I(operator.mul(self._unwrap(other), self.z3_var))

    # Comparison
    def __lt__(self, other):
        return Z3I(operator.lt(self.z3_var, self._unwrap(other)))

    def __le__(self, other):
        return Z3I(operator.le(self.z3_var, self._unwrap(other)))

    def __ne__(self, other):
        return Z3I(operator.ne(self.z3_var, self._unwrap(other)))

    def __gt__(self, other):
        return Z3I(operator.gt(self.z3_var, self._unwrap(other)))

    def __ge__(self, other):
        return Z3I(operator.ge(self.z3_var, self._unwrap(other)))

    def __eq__(self, other):
        return Z3I(operator.eq(self.z3_var, self._unwrap(other)))

    # Boolean operations
    def __and__(self, other):
        return Z3I(And(self.z3_var, self._unwrap(other)))

    def __or__(self, other):
        return Z3I(Or(self.z3_var, self._unwrap(other)))

    def __rshift__(self, other):
        return Z3I(Implies(self.z3_var, self._unwrap(other)))

    def __invert__(self):
        return Z3I(Not(self.z3_var))

    def not_(self):
        return Z3I(Not(self.z3_var))

    # ... add other methods as needed

    # ITE (If Then Else)
    @staticmethod
    def ite(if_cond, then_val, else_val):
        return Z3I(
            If(Z3I._unwrap(if_cond), Z3I._unwrap(then_val), Z3I._unwrap(else_val))
        )

    @staticmethod
    def cap(*args):
        unwrapped = [Z3I._unwrap(arg) for arg in args]
        return Z3I(And(unwrapped))

    @staticmethod
    def cup(*args):
        unwrapped = [Z3I._unwrap(arg) for arg in args]
        return Z3I(Or(unwrapped))


class STS:
    def __init__(self) -> None:
        self._state_variable_set = {}
        self._transition_list = []
        self._initial_condition_list = []
        self._safety_property_list = []
        self._invariant_list = []
        self._invariant_next_list = []

    @property
    def state_variable_set(self):
        return self._state_variable_set

    @property
    def transition_relation(self):
        unwrapped = [transition.z3_var for transition in self._transition_list]
        if len(unwrapped) == 0:
            return Z3I(True)
        elif len(unwrapped) == 1:
            return Z3I(unwrapped[0])
        else:
            return Z3I(And(unwrapped))

    @property
    def initial_condition(self):
        unwrapped = [initial.z3_var for initial in self._initial_condition_list]
        if len(unwrapped) == 0:
            return Z3I(True)
        elif len(unwrapped) == 1:
            return Z3I(unwrapped[0])
        else:
            return Z3I(And(unwrapped))

    @property
    def safety_property(self):
        unwrapped = [safety.z3_var for safety in self._safety_property_list]
        if len(unwrapped) == 0:
            return Z3I(True)
        elif len(unwrapped) == 1:
            return Z3I(unwrapped[0])
        else:
            return Z3I(And(unwrapped))

    @property
    def invariant(self):
        unwrapped = [inv.z3_var for inv in self._invariant_list]
        if len(unwrapped) == 0:
            return Z3I(True)
        elif len(unwrapped) == 1:
            return Z3I(unwrapped[0])
        else:
            return Z3I(And(unwrapped))

    @property
    def initial_implies_invariant(self):
        return self.initial_condition >> self.invariant

    @property
    def invariant_inductiveness(self):
        unwrapped_inv_next = [inv.z3_var for inv in self._invariant_next_list]
        if len(unwrapped_inv_next) == 0:
            return Z3I(True)
        elif len(unwrapped_inv_next) == 1:
            inv_next = Z3I(unwrapped_inv_next[0])
        else:
            inv_next = Z3I(And(unwrapped_inv_next))

        return (self.invariant & self.transition_relation) >> inv_next

    @property
    def invariant_implies_safety(self):
        return self.invariant >> self.safety_property

    def make_variable(self, name, type_initializer):
        if (
            name in self._state_variable_set
            or name + "__next" in self._state_variable_set
        ):
            raise Exception(
                f"Variable `{name}` already exists, or `{name}__next` has been already defined"
            )
        s = type_initializer(name)
        s_ = type_initializer(name + "__next")
        sI = Z3I(s)
        s_I = Z3I(s_)
        self._state_variable_set[name] = sI
        self._state_variable_set[name + "__next"] = s_I
        return self.state_variable_set[name], self.state_variable_set[name + "__next"]

    def make_transition_relation(self, func):
        def wrapper(*args, **kwargs):
            pred = func(*args, **kwargs)
            self._transition_list.append(pred)
            return pred

        return wrapper

    def make_initial_condition(self, func):
        def wrapper(*args, **kwargs):
            pred = func(*args, **kwargs)
            self._initial_condition_list.append(pred)
            return pred

        return wrapper

    def make_safety_property(self, func):
        def wrapper(*args, **kwargs):
            pred = func(*args, **kwargs)
            self._safety_property_list.append(pred)
            return pred

        return wrapper

    def make_invariant(self, func):
        def wrapper(*args, **kwargs):
            current_inv = func(*args, **kwargs)

            # Construct next state invariant
            subs = {
                self._state_variable_set[name]
                .z3_var: self._state_variable_set[name + "__next"]
                .z3_var
                for name in self._state_variable_set.keys()
                if not name.endswith("__next")
            }
            next_inv = Z3I(substitute(Z3I._unwrap(current_inv), *subs.items()))

            self._invariant_list.append(current_inv)
            self._invariant_next_list.append(next_inv)
            return current_inv  # Return current invariant for possible further use

        return wrapper


def sts_safety_check(
    sts,
    init_list,
    trans_list,
    inv_list,
    safety_list=[],
    only_check_inductive_invariant=False,
    print_smtlib2=True,
    print_sexpr_instead=False,
    print_statistics=True,
    verbose=True,
):
    # initialize predicates
    for f in init_list + trans_list + safety_list + inv_list:
        f()

    # set verbose
    if verbose:
        set_option(verbose=10)
    else:
        set_option(verbose=0)

    result_collection = []

    def check_z3_form_valid_then_print_result(form):
        s = Solver()
        s.add(Not(form))

        if print_smtlib2:
            if print_sexpr_instead:
                print("SMTLIB2 (sexpr):")
                print(s.sexpr())
            else:
                print("SMTLIB2:")
                print(s.to_smt2())
            print()

        result = s.check()
        result_collection.append(result)

        if print_statistics:
            stat = s.statistics()
            print(stat)

        print(f"reuslt: {result}")

        if result == sat:
            print("model (counter example):")
            print(s.model())

        print()
        return s.to_smt2()

    smt2_strs = []
    # check
    print("--------------------------------")
    print("Checking [init => invariant] ...")
    print("--------------------------------")
    form = sts.initial_implies_invariant.z3_var
    init2inv_smt2 = check_z3_form_valid_then_print_result(form)
    smt2_strs.append(init2inv_smt2)
    

    print("---------------------------------------------")
    print("Checking [(trans /\ invariant) => trans'] ...")
    print("---------------------------------------------")
    form = sts.invariant_inductiveness.z3_var
    trans_inv2next_smt2 = check_z3_form_valid_then_print_result(form)
    smt2_strs.append(trans_inv2next_smt2)

    if (not only_check_inductive_invariant) and safety_list:
        print("----------------------------------")
        print("Checking [invariant => safety] ...")
        print("----------------------------------")
        form = sts.invariant_implies_safety.z3_var
        inv2safety_smt2 = check_z3_form_valid_then_print_result(form)
        smt2_strs.append(inv2safety_smt2)


        print()
        if all([result == unsat for result in result_collection]):
            print("----------------------------------")
            print("STS statisfies the safety property")
            print("----------------------------------")
        else:
            print("-----------------------------------------")
            print("STS does NOT statisfy the safety property")
            print("Please check the counter example above")
            print("-----------------------------------------")

    return smt2_strs