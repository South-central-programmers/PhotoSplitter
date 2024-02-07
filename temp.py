import sqlite3

# Подключение к базе данных
conn = sqlite3.connect('src/backend/db.sqlite3')
c = conn.cursor()

# Удаление всех строк из таблицы
c.execute('DELETE FROM support_usersidentificationphotos;')

# Сохранение изменений
conn.commit()

# Закрытие соединения
conn.close()
