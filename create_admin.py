from app import app, db, bcrypt
from models import User

with app.app_context():
    # Check if admin already exists || Verificar si el admin ya existe
    existing_admin = User.query.filter_by(username='admin').first()
    
    if not existing_admin:
        # Hash the password for security || Encriptar la contraseña por seguridad
        hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
        
        # Create the Administrator user || Crear el usuario Administrador
        admin_user = User(username='admin', password_hash=hashed_pw, role='Admin')
        
        # Save to database || Guardar en la base de datos
        db.session.add(admin_user)
        db.session.commit()
        print("Success! User: admin | Password: admin123 || ¡Éxito! Usuario: admin | Clave: admin123")
    else:
        print("The administrator already exists || El administrador ya existe")