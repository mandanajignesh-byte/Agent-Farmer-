"""Model of the market price curve.

The competition documents the curve exactly:

    price(inv) = base + sign * amp * f(|inv - I0|)
    sign = +1 if inv < I0 (scarcity), -1 if inv > I0 (glut)
    amp  = target * base / f(T)

Selling pushes inventory up and price down, so what a tile is worth depends on
how much of that product we are about to add. A crop priced at today's quote is
priced as if we were not about to flood it - which is exactly the mistake that
made wheat look reasonable and strawberry look free.
"""
import math

I0 = 10_000

# base, T, below_func, below_target, above_func, above_target
PARAMS = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}

SHAPES = {
    "linear": lambda x, t: x,
    "sq": lambda x, t: x * x,
    "sqrt": lambda x, t: math.sqrt(x),
    "log": lambda x, t: math.log(1 + x),
    "log10": lambda x, t: math.log10(1 + x),
    "hinge": lambda x, t: (x / t) + 8 * max(0.0, x / t - 1) ** 2,
}


def price_at(product, inventory):
    """Price when market inventory sits at `inventory`. Scarcity raises it,
    glut lowers it, and the two sides use different curves - wheat barely sags
    on glut but spikes on scarcity, while melon does the opposite."""
    if product not in PARAMS:
        return 0
    base, t, below_func, below_target, above_func, above_target = PARAMS[product]
    gap = abs(inventory - I0)
    if gap == 0:
        return base

    scarce = inventory < I0
    func, target = (below_func, below_target) if scarce else (above_func, above_target)
    shape = SHAPES[func]
    amp = target * base / shape(t, t)
    move = amp * shape(gap, t)
    return max(1, round(base + move if scarce else base - move))


def revenue_for(product, inventory, units):
    """Total takings for selling `units`, priced one at a time as the price
    falls. This is what a marginal tile is actually worth, not units * quote."""
    total = 0
    for i in range(int(units)):
        total += price_at(product, inventory + i)
    return total
