"""
create_admin.py
----------------------------------------------------------------------
One-time helper script to create an administrator account.

This is intentionally a separate script, NOT a Flask route, because
admin accounts should never be creatable through the public-facing
/register form -- otherwise any visitor could grant themselves
admin access by simply submitting role=admin in a form.

Run with:   python create_admin.py
You will be prompted for a username, email, and password.
----------------------------------------------------------------------
"""

import sys
import getpass

sys.path.insert(0, ".")
from app.models import User
from app.database import init_db


def main():
    init_db()

    print("=== Create Administrator Account ===\n")
    username = input("Admin username: ").strip()
    email    = input("Admin email: ").strip()
    password = getpass.getpass("Admin password: ")
    confirm  = getpass.getpass("Confirm password: ")

    if not username or not email or not password:
        print("\nAll fields are required. Aborting.")
        return

    if password != confirm:
        print("\nPasswords do not match. Aborting.")
        return

    if User.get_by_username(username) is not None:
        print(f"\nA user named '{username}' already exists. Aborting.")
        return

    User.create(username, email, password, role="admin")
    print(f"\nAdministrator account '{username}' created successfully.")


if __name__ == "__main__":
    main()
