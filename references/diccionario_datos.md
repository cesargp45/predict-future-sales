# Diccionario de datos — Predict Future Sales

Material de referencia del proyecto. Las traducciones provienen de las constantes
`CATEGORY_NAMES_ES` y `SHOP_CITY_ES` de `predict_future_sales/dataset.py` (este archivo se generó a partir de ellas).

## Columnas

| Columna | Archivo | Tipo | Descripción |
|---|---|---|---|
| `date` | sales_train | fecha | Día de la transacción (texto `dd.mm.aaaa` en el original) |
| `date_block_num` | sales_train | ordinal | Mes consecutivo: 0 = ene-2013 … 33 = oct-2015 (34 = nov-2015, mes a predecir) |
| `shop_id` | sales_train, shops, test | categórica | Identificador de la tienda |
| `item_id` | sales_train, items, test | categórica | Identificador del producto |
| `item_price` | sales_train | continua | Precio unitario en rublos |
| `item_cnt_day` | sales_train | discreta | Unidades vendidas en el día (negativo = devolución) |
| `item_name` | items | texto | Nombre del producto (ruso) |
| `item_category_id` | items, item_categories | categórica | Categoría del producto |
| `item_category_name` | item_categories | texto | Nombre de la categoría `Grupo - Subgrupo` (ruso) |
| `shop_name` | shops | texto | Ciudad + centro comercial (ruso) |
| `ID` | test | entero | Fila a predecir (par tienda–producto) |

**Objetivo:** `item_cnt_month` = suma mensual de `item_cnt_day` por par tienda–producto, recortada a [0, 20] por Kaggle.

**Derivadas (EDA):** `revenue = item_price × item_cnt_day`, `is_return`, `year`, `month`, `day_of_week`, `category_es`, `category_group`, `city`, `shop_type`, `shop_label`.

## Tiendas duplicadas unificadas

| ID antiguo | ID vigente |
|---|---|
| 0 | 57 |
| 1 | 58 |
| 11 | 10 |

## Categorías (84)

| ID | Nombre original (ruso) | Traducción | Grupo |
|---|---|---|---|
| 0 | PC - Гарнитуры/Наушники | PC - Auriculares/Headsets | PC |
| 1 | Аксессуары - PS2 | Accesorios - PS2 | Accesorios |
| 2 | Аксессуары - PS3 | Accesorios - PS3 | Accesorios |
| 3 | Аксессуары - PS4 | Accesorios - PS4 | Accesorios |
| 4 | Аксессуары - PSP | Accesorios - PSP | Accesorios |
| 5 | Аксессуары - PSVita | Accesorios - PSVita | Accesorios |
| 6 | Аксессуары - XBOX 360 | Accesorios - XBOX 360 | Accesorios |
| 7 | Аксессуары - XBOX ONE | Accesorios - XBOX ONE | Accesorios |
| 8 | Билеты (Цифра) | Entradas (Digital) | Entradas |
| 9 | Доставка товара | Envío de productos | Envío de productos |
| 10 | Игровые консоли - PS2 | Consolas - PS2 | Consolas |
| 11 | Игровые консоли - PS3 | Consolas - PS3 | Consolas |
| 12 | Игровые консоли - PS4 | Consolas - PS4 | Consolas |
| 13 | Игровые консоли - PSP | Consolas - PSP | Consolas |
| 14 | Игровые консоли - PSVita | Consolas - PSVita | Consolas |
| 15 | Игровые консоли - XBOX 360 | Consolas - XBOX 360 | Consolas |
| 16 | Игровые консоли - XBOX ONE | Consolas - XBOX ONE | Consolas |
| 17 | Игровые консоли - Прочие | Consolas - Otras | Consolas |
| 18 | Игры - PS2 | Juegos - PS2 | Juegos |
| 19 | Игры - PS3 | Juegos - PS3 | Juegos |
| 20 | Игры - PS4 | Juegos - PS4 | Juegos |
| 21 | Игры - PSP | Juegos - PSP | Juegos |
| 22 | Игры - PSVita | Juegos - PSVita | Juegos |
| 23 | Игры - XBOX 360 | Juegos - XBOX 360 | Juegos |
| 24 | Игры - XBOX ONE | Juegos - XBOX ONE | Juegos |
| 25 | Игры - Аксессуары для игр | Juegos - Accesorios para juegos | Juegos |
| 26 | Игры Android - Цифра | Juegos Android - Digital | Juegos Android |
| 27 | Игры MAC - Цифра | Juegos MAC - Digital | Juegos MAC |
| 28 | Игры PC - Дополнительные издания | Juegos PC - Ediciones adicionales | Juegos PC |
| 29 | Игры PC - Коллекционные издания | Juegos PC - Ediciones de colección | Juegos PC |
| 30 | Игры PC - Стандартные издания | Juegos PC - Ediciones estándar | Juegos PC |
| 31 | Игры PC - Цифра | Juegos PC - Digital | Juegos PC |
| 32 | Карты оплаты (Кино, Музыка, Игры) | Tarjetas de pago (Cine, Música, Juegos) | Tarjetas de pago |
| 33 | Карты оплаты - Live! | Tarjetas de pago - Live! | Tarjetas de pago |
| 34 | Карты оплаты - Live! (Цифра) | Tarjetas de pago - Live! (Digital) | Tarjetas de pago |
| 35 | Карты оплаты - PSN | Tarjetas de pago - PSN | Tarjetas de pago |
| 36 | Карты оплаты - Windows (Цифра) | Tarjetas de pago - Windows (Digital) | Tarjetas de pago |
| 37 | Кино - Blu-Ray | Cine - Blu-Ray | Cine |
| 38 | Кино - Blu-Ray 3D | Cine - Blu-Ray 3D | Cine |
| 39 | Кино - Blu-Ray 4K | Cine - Blu-Ray 4K | Cine |
| 40 | Кино - DVD | Cine - DVD | Cine |
| 41 | Кино - Коллекционное | Cine - Coleccionables | Cine |
| 42 | Книги - Артбуки, энциклопедии | Libros - Artbooks, enciclopedias | Libros |
| 43 | Книги - Аудиокниги | Libros - Audiolibros | Libros |
| 44 | Книги - Аудиокниги (Цифра) | Libros - Audiolibros (Digital) | Libros |
| 45 | Книги - Аудиокниги 1С | Libros - Audiolibros 1C | Libros |
| 46 | Книги - Бизнес литература | Libros - Negocios | Libros |
| 47 | Книги - Комиксы, манга | Libros - Cómics, manga | Libros |
| 48 | Книги - Компьютерная литература | Libros - Computación | Libros |
| 49 | Книги - Методические материалы 1С | Libros - Material metodológico 1C | Libros |
| 50 | Книги - Открытки | Libros - Postales | Libros |
| 51 | Книги - Познавательная литература | Libros - Divulgación | Libros |
| 52 | Книги - Путеводители | Libros - Guías de viaje | Libros |
| 53 | Книги - Художественная литература | Libros - Ficción | Libros |
| 54 | Книги - Цифра | Libros - Digital | Libros |
| 55 | Музыка - CD локального производства | Música - CD producción local | Música |
| 56 | Музыка - CD фирменного производства | Música - CD producción de marca | Música |
| 57 | Музыка - MP3 | Música - MP3 | Música |
| 58 | Музыка - Винил | Música - Vinilo | Música |
| 59 | Музыка - Музыкальное видео | Música - Video musical | Música |
| 60 | Музыка - Подарочные издания | Música - Ediciones de regalo | Música |
| 61 | Подарки - Атрибутика | Regalos - Artículos promocionales | Regalos |
| 62 | Подарки - Гаджеты, роботы, спорт | Regalos - Gadgets, robots, deporte | Regalos |
| 63 | Подарки - Мягкие игрушки | Regalos - Peluches | Regalos |
| 64 | Подарки - Настольные игры | Regalos - Juegos de mesa | Regalos |
| 65 | Подарки - Настольные игры (компактные) | Regalos - Juegos de mesa (compactos) | Regalos |
| 66 | Подарки - Открытки, наклейки | Regalos - Postales, stickers | Regalos |
| 67 | Подарки - Развитие | Regalos - Didácticos | Regalos |
| 68 | Подарки - Сертификаты, услуги | Regalos - Certificados, servicios | Regalos |
| 69 | Подарки - Сувениры | Regalos - Souvenirs | Regalos |
| 70 | Подарки - Сувениры (в навеску) | Regalos - Souvenirs (colgantes) | Regalos |
| 71 | Подарки - Сумки, Альбомы, Коврики д/мыши | Regalos - Bolsas, álbumes, mousepads | Regalos |
| 72 | Подарки - Фигурки | Regalos - Figuras | Regalos |
| 73 | Программы - 1С:Предприятие 8 | Programas - 1C:Empresa 8 | Programas |
| 74 | Программы - MAC (Цифра) | Programas - MAC (Digital) | Programas |
| 75 | Программы - Для дома и офиса | Programas - Hogar y oficina | Programas |
| 76 | Программы - Для дома и офиса (Цифра) | Programas - Hogar y oficina (Digital) | Programas |
| 77 | Программы - Обучающие | Programas - Educativos | Programas |
| 78 | Программы - Обучающие (Цифра) | Programas - Educativos (Digital) | Programas |
| 79 | Служебные | Servicios | Servicios |
| 80 | Служебные - Билеты | Servicios - Entradas | Servicios |
| 81 | Чистые носители (шпиль) | Medios vírgenes (spindle) | Medios vírgenes |
| 82 | Чистые носители (штучные) | Medios vírgenes (unidad) | Medios vírgenes |
| 83 | Элементы питания | Baterías | Baterías |

## Tiendas (60)

| ID | Nombre original (ruso) | Ciudad | Tipo |
|---|---|---|---|
| 0 | !Якутск Орджоникидзе, 56 фран | Yakutsk | Calle |
| 1 | !Якутск ТЦ "Центральный" фран | Yakutsk | Centro comercial |
| 2 | Адыгея ТЦ "Мега" | Adigueya | Centro comercial |
| 3 | Балашиха ТРК "Октябрь-Киномир" | Balashikha | Centro comercial |
| 4 | Волжский ТЦ "Волга Молл" | Volzhski | Centro comercial |
| 5 | Вологда ТРЦ "Мармелад" | Vólogda | Centro comercial |
| 6 | Воронеж (Плехановская, 13) | Vorónezh | Calle |
| 7 | Воронеж ТРЦ "Максимир" | Vorónezh | Centro comercial |
| 8 | Воронеж ТРЦ Сити-Парк "Град" | Vorónezh | Centro comercial |
| 9 | Выездная Торговля | Venta itinerante | Itinerante |
| 10 | Жуковский ул. Чкалова 39м? | Zhukovski | Calle |
| 11 | Жуковский ул. Чкалова 39м² | Zhukovski | Calle |
| 12 | Интернет-магазин ЧС | Tienda online | Online |
| 13 | Казань ТЦ "Бехетле" | Kazán | Centro comercial |
| 14 | Казань ТЦ "ПаркХаус" II | Kazán | Centro comercial |
| 15 | Калуга ТРЦ "XXI век" | Kaluga | Centro comercial |
| 16 | Коломна ТЦ "Рио" | Kolomna | Centro comercial |
| 17 | Красноярск ТЦ "Взлетка Плаза" | Krasnoyarsk | Centro comercial |
| 18 | Красноярск ТЦ "Июнь" | Krasnoyarsk | Centro comercial |
| 19 | Курск ТЦ "Пушкинский" | Kursk | Centro comercial |
| 20 | Москва "Распродажа" | Moscú | Liquidación |
| 21 | Москва МТРЦ "Афи Молл" | Moscú | Centro comercial |
| 22 | Москва Магазин С21 | Moscú | Calle |
| 23 | Москва ТК "Буденовский" (пав.А2) | Moscú | Centro comercial |
| 24 | Москва ТК "Буденовский" (пав.К7) | Moscú | Centro comercial |
| 25 | Москва ТРК "Атриум" | Moscú | Centro comercial |
| 26 | Москва ТЦ "Ареал" (Беляево) | Moscú | Centro comercial |
| 27 | Москва ТЦ "МЕГА Белая Дача II" | Moscú | Centro comercial |
| 28 | Москва ТЦ "МЕГА Теплый Стан" II | Moscú | Centro comercial |
| 29 | Москва ТЦ "Новый век" (Новокосино) | Moscú | Centro comercial |
| 30 | Москва ТЦ "Перловский" | Moscú | Centro comercial |
| 31 | Москва ТЦ "Семеновский" | Moscú | Centro comercial |
| 32 | Москва ТЦ "Серебряный Дом" | Moscú | Centro comercial |
| 33 | Мытищи ТРК "XL-3" | Mytishchi | Centro comercial |
| 34 | Н.Новгород ТРЦ "РИО" | N. Nóvgorod | Centro comercial |
| 35 | Н.Новгород ТРЦ "Фантастика" | N. Nóvgorod | Centro comercial |
| 36 | Новосибирск ТРЦ "Галерея Новосибирск" | Novosibirsk | Centro comercial |
| 37 | Новосибирск ТЦ "Мега" | Novosibirsk | Centro comercial |
| 38 | Омск ТЦ "Мега" | Omsk | Centro comercial |
| 39 | РостовНаДону ТРК "Мегацентр Горизонт" | Rostov del Don | Centro comercial |
| 40 | РостовНаДону ТРК "Мегацентр Горизонт" Островной | Rostov del Don | Centro comercial |
| 41 | РостовНаДону ТЦ "Мега" | Rostov del Don | Centro comercial |
| 42 | СПб ТК "Невский Центр" | San Petersburgo | Centro comercial |
| 43 | СПб ТК "Сенная" | San Petersburgo | Centro comercial |
| 44 | Самара ТЦ "Мелодия" | Samara | Centro comercial |
| 45 | Самара ТЦ "ПаркХаус" | Samara | Centro comercial |
| 46 | Сергиев Посад ТЦ "7Я" | Sérguiev Posad | Centro comercial |
| 47 | Сургут ТРЦ "Сити Молл" | Surgut | Centro comercial |
| 48 | Томск ТРЦ "Изумрудный Город" | Tomsk | Centro comercial |
| 49 | Тюмень ТРЦ "Кристалл" | Tiumén | Centro comercial |
| 50 | Тюмень ТЦ "Гудвин" | Tiumén | Centro comercial |
| 51 | Тюмень ТЦ "Зеленый Берег" | Tiumén | Centro comercial |
| 52 | Уфа ТК "Центральный" | Ufá | Centro comercial |
| 53 | Уфа ТЦ "Семья" 2 | Ufá | Centro comercial |
| 54 | Химки ТЦ "Мега" | Jimki | Centro comercial |
| 55 | Цифровой склад 1С-Онлайн | Almacén digital | Online |
| 56 | Чехов ТРЦ "Карнавал" | Chéjov | Centro comercial |
| 57 | Якутск Орджоникидзе, 56 | Yakutsk | Calle |
| 58 | Якутск ТЦ "Центральный" | Yakutsk | Centro comercial |
| 59 | Ярославль ТЦ "Альтаир" | Yaroslavl | Centro comercial |
