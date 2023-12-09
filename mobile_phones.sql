-- Tabel untuk menyimpan spesifikasi layar
CREATE TABLE display_specs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    display_type VARCHAR(255),
    display_resolution VARCHAR(255),
    display_size DECIMAL(4,2)
);

-- Tabel untuk menyimpan informasi performa
CREATE TABLE performance_specs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    CPU VARCHAR(255),
    RAM INT,
    internal_memory INT
);

-- Tabel untuk menyimpan informasi kamera
CREATE TABLE camera_specs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    primary_camera VARCHAR(255),
    secondary_camera VARCHAR(255)
);

-- Tabel untuk menyimpan informasi kriteria dan bobotnya
CREATE TABLE criteria_weights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    criterion_name VARCHAR(255),
    weight DECIMAL(3,2)
);

-- Tabel untuk menyimpan informasi ponsel
CREATE TABLE smartphones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    brand VARCHAR(255),
    model VARCHAR(255),
    price_EUR INT,
    battery VARCHAR(255),
    dimensions VARCHAR(255),
    weight_g INT,
    img_url VARCHAR(255),
    display_specs_id INT,
    performance_specs_id INT,
    camera_specs_id INT,
    FOREIGN KEY (display_specs_id) REFERENCES display_specs(id),
    FOREIGN KEY (performance_specs_id) REFERENCES performance_specs(id),
    FOREIGN KEY (camera_specs_id) REFERENCES camera_specs(id)
);

-- Tabel untuk menyimpan skor atau nilai SAW untuk setiap ponsel
CREATE TABLE smartphone_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    smartphone_id INT,
    criterion_id INT,
    score DECIMAL(10,2),
    FOREIGN KEY (smartphone_id) REFERENCES smartphones(id),
    FOREIGN KEY (criterion_id) REFERENCES criteria_weights(id)
);
