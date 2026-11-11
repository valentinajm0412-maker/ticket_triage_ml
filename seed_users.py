from app import app, db, bcrypt
from models import User

# Script to populate initial test users || Script para poblar usuarios iniciales de prueba
with app.app_context():
    test_users = [
        {"username": "tecnico1", "role": "Consultor Tecnico"},
        {"username": "tecnico2", "role": "Consultor Tecnico"},
        {"username": "funcional1", "role": "Consultor Funcional"},
        {"username": "funcional2", "role": "Consultor Funcional"},
        {"username": "cliente1", "role": "Cliente"}
    ]
    
    # Default password for testing || Contraseña por defecto para pruebas
    default_pw_hash = bcrypt.generate_password_hash("123456").decode('utf-8')
    
    for u in test_users:
        # Check if user already exists || Verificar si el usuario ya existe
        exists = User.query.filter_by(username=u["username"]).first()
        if not exists:
            new_user = User(
                username=u["username"],
                password_hash=default_pw_hash,
                role=u["role"],
                workload=0
            )
            db.session.add(new_user)
            print(f"Created user: {u['username']} ({u['role']}) || Usuario creado: {u['username']} ({u['role']})")
            
    db.session.commit()
    print("\nInitial users successfully added! || ¡Usuarios iniciales agregados con éxito!")