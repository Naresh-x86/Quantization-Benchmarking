import json
import inspect
import copy

# --- INITIAL STATE (used to reset between trials) ---
_INITIAL_ORDERS = {
    "ORD-992": {
        "user_id": 1045,
        "item_id": "ITEM-ZB-PRO",
        "item_name": "ZenBook Pro",
        "status": "delivered",
        "price": 1499.99
    }
}
_INITIAL_INVENTORY = {
    "ITEM-ZB-PRO": {
        "in_stock": True,
        "shipping_days": 5
    }
}
_INITIAL_SERVICES = {
    "api_gateway": "online",
    "auth_service": "online",
    "payment_backend": "down",
    "database": "online"
}
_INITIAL_LOGS = {
    "payment_backend": "ERROR: Connection refused to database port 5432. Process crashed."
}
_INITIAL_STOCK_DATA = {
    "TECH_CORP": {
        "current_price": 145.50,
        "5_day_history": [152.0, 150.5, 149.0, 147.5, 145.5]
    }
}

# --- MUTABLE STATE (reset before each trial) ---
ORDERS = copy.deepcopy(_INITIAL_ORDERS)
INVENTORY = copy.deepcopy(_INITIAL_INVENTORY)
SERVICES = copy.deepcopy(_INITIAL_SERVICES)
LOGS = copy.deepcopy(_INITIAL_LOGS)
STOCK_DATA = copy.deepcopy(_INITIAL_STOCK_DATA)

def reset_tool_state():
    """Reset all mutable global state to initial values. Call this before each trial."""
    global ORDERS, INVENTORY, SERVICES, LOGS, STOCK_DATA
    ORDERS = copy.deepcopy(_INITIAL_ORDERS)
    INVENTORY = copy.deepcopy(_INITIAL_INVENTORY)
    SERVICES = copy.deepcopy(_INITIAL_SERVICES)
    LOGS = copy.deepcopy(_INITIAL_LOGS)
    STOCK_DATA = copy.deepcopy(_INITIAL_STOCK_DATA)

# --- AGENT 1: CUSTOMER SUPPORT TOOLS ---
def check_order(order_id: str) -> str:
    """Look up an order by its ID. Returns order details including user_id, item_id, item_name, status, and price."""
    if order_id in ORDERS:
        return json.dumps(ORDERS[order_id])
    return "Order not found."

def check_inventory(item_id: str) -> str:
    """Check inventory for an item by its item_id. Returns stock availability and estimated shipping_days."""
    if item_id in INVENTORY:
        return json.dumps(INVENTORY[item_id])
    return "Item not found in inventory."

def issue_replacement(order_id: str) -> str:
    """Issue a replacement for the given order_id."""
    if order_id in ORDERS:
        return f"Replacement issued for order {order_id}."
    return "Order not found."

def issue_refund(order_id: str) -> str:
    """Issue a full refund for the given order_id."""
    if order_id in ORDERS:
        return f"Refund of ${ORDERS[order_id]['price']} issued for order {order_id}."
    return "Order not found."

# --- AGENT 2: IT HELPDESK TOOLS ---
def check_server_status() -> str:
    """Check the status of all services. Returns a JSON object mapping service names to their status (online/down)."""
    return json.dumps(SERVICES)

def read_service_logs(service_name: str) -> str:
    """Read the logs for a specific service. Requires the service_name as a string."""
    if service_name in LOGS:
        return LOGS[service_name]
    return f"No logs found for {service_name}."

def restart_service(service_name: str) -> str:
    """Restart a specific service by its service_name."""
    if service_name in SERVICES:
        SERVICES[service_name] = "online"
        return f"Service {service_name} has been successfully restarted."
    return "Service not found."

# --- AGENT 3: FINANCIAL ANALYST TOOLS ---
def get_current_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    if ticker in STOCK_DATA:
        return f"Current price of {ticker} is ${STOCK_DATA[ticker]['current_price']}"
    return "Ticker not found."

def get_5_day_average(ticker: str) -> str:
    """Calculate the 5-day average stock price for a given ticker symbol."""
    if ticker in STOCK_DATA:
        history = STOCK_DATA[ticker]["5_day_history"]
        avg = sum(history) / len(history)
        return f"5-day average for {ticker} is ${avg:.2f}"
    return "Ticker not found."

def execute_trade(action: str, ticker: str, shares: int) -> str:
    """Execute a stock trade. action must be 'buy' or 'sell'. ticker is the stock symbol. shares is the number of shares."""
    if action not in ["buy", "sell"]:
        return "Invalid action. Must be 'buy' or 'sell'."
    if ticker in STOCK_DATA:
        return f"Successfully executed {action} of {shares} shares for {ticker}."
    return "Ticker not found."

# --- TOOL REGISTRIES ---
SUPPORT_TOOLS = {
    "check_order": check_order,
    "check_inventory": check_inventory,
    "issue_replacement": issue_replacement,
    "issue_refund": issue_refund
}

IT_TOOLS = {
    "check_server_status": check_server_status,
    "read_service_logs": read_service_logs,
    "restart_service": restart_service
}

FINANCE_TOOLS = {
    "get_current_price": get_current_price,
    "get_5_day_average": get_5_day_average,
    "execute_trade": execute_trade
}

def get_tools_description(agent_id: str) -> str:
    """Generate a detailed tool description including function signatures and docstrings."""
    tools_dict = get_tools_dict(agent_id)
    desc = "Available tools:\n"
    for name, func in tools_dict.items():
        sig = inspect.signature(func)
        params = ", ".join(
            f"{p.name}: {p.annotation.__name__}" if p.annotation != inspect.Parameter.empty else p.name
            for p in sig.parameters.values()
        )
        docstring = func.__doc__ or "No description available."
        desc += f"- {name}({params}): {docstring}\n"
    return desc

def get_tools_dict(agent_id: str) -> dict:
    if agent_id == "AGENT_1_SUPPORT":
        return SUPPORT_TOOLS
    elif agent_id == "AGENT_2_IT_HELPDESK":
        return IT_TOOLS
    elif agent_id == "AGENT_3_FINANCE":
        return FINANCE_TOOLS
    return {}
