class User:
    def __init__(self, id: str = "demo-user"):
        self.id = id

def get_current_user() -> "User":
    # TODO: Replace with real auth (JWT/session)
    return User()
