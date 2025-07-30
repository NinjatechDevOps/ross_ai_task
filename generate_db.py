
import sqlite3
import random

def generate_products_db():
    """
    Generates a products.db SQLite database with 50 mock laptop entries.
    """
    db_file = "products.db"
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create the laptops table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS laptops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            name TEXT NOT NULL,
            processor TEXT NOT NULL,
            ram_gb INTEGER NOT NULL,
            storage_gb INTEGER NOT NULL,
            screen_size_inches REAL NOT NULL,
            price_usd REAL NOT NULL
        );
        """)

        # --- Mock Data ---
        brands = ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "Razer", "MSI"]
        models = ["Inspiron", "XPS", "ThinkPad", "MacBook Pro", "MacBook Air", "ZenBook", "ROG", "Predator", "Blade"]
        processors = ["Intel Core i5", "Intel Core i7", "Intel Core i9", "AMD Ryzen 5", "AMD Ryzen 7", "Apple M2", "Apple M3 Pro"]
        ram_options = [8, 16, 32, 64]
        storage_options = [256, 512, 1024, 2048]
        screen_sizes = [13.3, 14.0, 15.6, 16.0, 17.3]

        laptops_to_add = 50
        for i in range(laptops_to_add):
            brand = random.choice(brands)
            name = f"{random.choice(models)} {random.randint(100, 999)}"
            
            # Make Apple laptops have Apple processors
            if brand == "Apple":
                processor = random.choice([p for p in processors if "Apple" in p])
                name = random.choice(["MacBook Pro", "MacBook Air"])
            else:
                processor = random.choice([p for p in processors if "Apple" not in p])

            ram = random.choice(ram_options)
            storage = random.choice(storage_options)
            screen = random.choice(screen_sizes)
            
            # Generate a realistic price based on specs
            price = (
                500
                + (ram * 20)
                + (storage * 0.2)
                + (150 if "i7" in processor or "Ryzen 7" in processor else 0)
                + (350 if "i9" in processor else 0)
                + (400 if "Apple" in processor else 0)
                + random.uniform(-50, 50) # Add some noise
            )

            cursor.execute("""
            INSERT INTO laptops (brand, name, processor, ram_gb, storage_gb, screen_size_inches, price_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (brand, name, processor, ram, storage, screen, round(price, 2)))

        conn.commit()
        print(f"Successfully created '{db_file}' with {laptops_to_add} laptop entries.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    generate_products_db()
