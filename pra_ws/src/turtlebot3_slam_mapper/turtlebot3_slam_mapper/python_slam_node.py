#!/usr/bin/env python3
"""
Nodo de SLAM usando un filtro de partículas (Particle Filter SLAM / FastSLAM)

Este nodo implementa un algoritmo de SLAM (Simultaneous Localization and Mapping)
que permite al robot construir un mapa del entorno mientras estima su propia posición.
Utiliza un filtro de partículas donde cada partícula mantiene su propio mapa y pose estimada.

Autor: Equipo de Robótica
"""

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from scipy.spatial.transform import Rotation as R
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

def normalize_angle(angle):
    """
    Normaliza un ángulo al rango [-π, π] de forma robusta
    
    Esta función es fundamental para evitar problemas de discontinuidad
    en ángulos cuando pasan de 2π a 0 o viceversa.
    
    Args:
        angle: ángulo en radianes
    Returns:
        float: ángulo normalizado en [-π, π]
    """
    if math.isnan(angle) or math.isinf(angle):
        return 0.0
    
    # Usar fmod para mayor precisión que operaciones manuales
    result = math.fmod(angle + math.pi, 2.0 * math.pi)
    if result < 0:
        result += 2.0 * math.pi
    return result - math.pi

def euler_from_quaternion(x, y, z, w):
    """
    Convierte un quaternion a ángulos de Euler (roll, pitch, yaw) de forma robusta
    
    Los quaternions son útiles para representar rotaciones sin problemas de gimbal lock,
    pero para el control del robot necesitamos el ángulo yaw (rotación en Z).
    
    Args:
        x, y, z, w: componentes del quaternion
    Returns:
        tuple: (roll, pitch, yaw) en radianes
    """
    # Normalizar quaternion para evitar errores numéricos
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-10:
        return 0.0, 0.0, 0.0
    
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Roll (x-axis rotation) - más robusto numéricamente
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation) - evitar gimbal lock
    sinp = 2.0 * (w * y - z * x)
    # Clamp para evitar errores numéricos
    sinp = max(-1.0, min(1.0, sinp))
    if abs(sinp) >= 0.99999:  # Casi gimbal lock
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation) - este es el más importante para robots terrestres
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

class Particle:
    """
    Representa una partícula individual en el filtro de partículas
    
    Cada partícula mantiene:
    - Su pose estimada (x, y, theta)
    - Su peso (probabilidad de ser la pose correcta)
    - Su propio mapa en log-odds
    - Su odometría anterior (para FastSLAM)
    """
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x              # Posición x en el mapa
        self.y = y              # Posición y en el mapa
        self.theta = theta      # Orientación (yaw) en radianes
        self.weight = weight    # Peso de la partícula (probabilidad)
        
        # Cada partícula mantiene su propio mapa en formato log-odds
        # log-odds > 0 = celda ocupada, log-odds < 0 = celda libre
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)
        
        # Almacenar la odometría anterior de esta partícula para FastSLAM
        # Esto permite calcular el movimiento relativo de cada partícula
        self.prev_odom_x = 0.0
        self.prev_odom_y = 0.0
        self.prev_odom_theta = 0.0

    def pose(self):
        """Retorna la pose actual como array numpy"""
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node):
    """
    Nodo principal de SLAM con filtro de partículas
    
    Implementa el algoritmo FastSLAM que combina:
    1. Filtro de partículas para localización
    2. Mapeo individual por partícula
    3. Modelo de movimiento con ruido
    4. Modelo de observación usando laser scan
    """
    
    def __init__(self):
        super().__init__('python_slam_node')

        # === PARÁMETROS DE CONFIGURACIÓN ===
        # Declarar parámetros ROS2 con valores por defecto
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        
        # Parámetros del mapa y filtro de partículas
        self.declare_parameter('map_resolution', 0.05)          # Resolución del mapa en metros/celda
        self.declare_parameter('map_width_meters', 6.0)         # Ancho del mapa en metros
        self.declare_parameter('map_height_meters', 6.0)        # Alto del mapa en metros
        self.declare_parameter('num_particles', 10)             # Número de partículas

        # Obtener valores de parámetros
        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        
        # Convertir dimensiones del mapa de metros a celdas
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        
        # Origen del mapa (esquina inferior izquierda) en coordenadas del mundo
        self.map_origin_x = -2.5
        self.map_origin_y = -5.0

        # === PARÁMETROS DE MAPEO ===
        # Valores de log-odds para actualización del mapa
        # Valores más pequeños = mapeo más conservador y estable
        self.log_odds_free = -0.05      # Incremento para celdas libres
        self.log_odds_occupied = 0.10   # Incremento para celdas ocupadas
        self.log_odds_max = 10.0        # Límite superior de log-odds
        self.log_odds_min = -10.0       # Límite inferior de log-odds

        # === INICIALIZACIÓN DEL FILTRO DE PARTÍCULAS ===
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        
        # Crear partículas iniciales todas en el origen con peso uniforme
        self.particles = [
            Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, 
                    (self.map_height_cells, self.map_width_cells)) 
            for _ in range(self.num_particles)
        ]
        
        # Variables de estado
        self.last_odom = None                           # Última odometría recibida
        self.current_map_pose = [0.0, 0.0, 0.0]       # Pose estimada en frame del mapa
        self.current_odom_pose = [0.0, 0.0, 0.0]      # Pose actual en frame de odometría

        # === PARÁMETROS DEL MODELO DE MOVIMIENTO ===
        # Parámetros de ruido del modelo de movimiento (valores reducidos para mayor estabilidad)
        self.alpha1 = 0.005  # ruido rotación -> rotación
        self.alpha2 = 0.005  # ruido traslación -> rotación  
        self.alpha3 = 0.005  # ruido traslación -> traslación
        self.alpha4 = 0.005  # ruido rotación -> traslación
        
        # === CONFIGURACIÓN DE ROS2 ===
        # Configurar QoS para el mapa (debe ser persistent para rviz)
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Publishers
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # === MANEJO DE TRANSFORMACIONES TF ===
        self.tf_buffer = Buffer()  # Buffer para almacenar transformaciones recientes
        self.tf_listener = TransformListener(self.tf_buffer, self)  # Escucha transformaciones del árbol TF
        
        # Subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10)
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        # Timer para publicar el mapa periódicamente
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info("Nodo de SLAM con filtro de partículas inicializado correctamente.")

    def odom_callback(self, msg: Odometry):
        """
        Callback para recibir datos de odometría
        
        La odometría nos da la pose estimada del robot basada en encoders de ruedas.
        Es útil para el modelo de movimiento pero acumula error con el tiempo.
        
        Args:
            msg: Mensaje de odometría con pose y twist
        """
        # Almacenar odometría para actualización de movimiento
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        """
        Callback principal que procesa laser scans y ejecuta el algoritmo de SLAM
        
        Este es el corazón del algoritmo FastSLAM:
        1. Actualización de movimiento (motion update)
        2. Actualización de medición (measurement update) 
        3. Remuestreo de partículas
        4. Estimación de pose promedio ponderada
        5. Actualización del mapa
        6. Publicación de transformaciones
        
        Args:
            msg: Mensaje de LaserScan con datos del LIDAR
        """
        if self.last_odom is None:
            return

        # === 1. ACTUALIZACIÓN DE MOVIMIENTO ===
        # Extraer pose de odometría actual
        odom = self.last_odom
        odom_x = odom.pose.pose.position.x
        odom_y = odom.pose.pose.position.y
        quaternion = odom.pose.pose.orientation
        _, _, odom_theta = euler_from_quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.current_odom_pose = [odom_x, odom_y, odom_theta]

        # FastSLAM: Para cada partícula, calcular movimiento desde SU odometría anterior
        if odom is not None:
            for p in self.particles:
                # Calcular diferencia desde la odometría anterior de ESTA partícula
                delta_x = odom_x - p.prev_odom_x
                delta_y = odom_y - p.prev_odom_y
                delta_theta = self.angle_diff(odom_theta, p.prev_odom_theta)
                
                # Aplicar modelo de movimiento con ruido gaussiano
                noise_x = np.random.normal(0, self.alpha3 * abs(delta_x) + self.alpha4 * abs(delta_theta))
                noise_y = np.random.normal(0, self.alpha3 * abs(delta_y) + self.alpha4 * abs(delta_theta))
                noise_theta = np.random.normal(0, self.alpha1 * abs(delta_theta) + self.alpha2 * (abs(delta_x) + abs(delta_y)))
                
                # Actualizar pose de la partícula
                p.x += delta_x + noise_x
                p.y += delta_y + noise_y
                p.theta = normalize_angle(p.theta + delta_theta + noise_theta)
                
                # Actualizar la odometría anterior de esta partícula
                p.prev_odom_x = odom_x
                p.prev_odom_y = odom_y
                p.prev_odom_theta = odom_theta

        # === 2. ACTUALIZACIÓN DE MEDICIÓN (CÁLCULO DE PESOS) ===
        weights = []
        for p in self.particles:
            # Calcular qué tan bien coincide el scan actual con el mapa de esta partícula
            weight = self.compute_weight(p, msg)
            weights.append(weight)

        # Normalizar pesos para que sumen 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Si no hay peso válido, distribución uniforme
            weights = [1.0 / len(self.particles) for _ in range(len(self.particles))]

        # Asignar pesos normalizados a las partículas
        for i, p in enumerate(self.particles):
            p.weight = weights[i]

        # === 3. REMUESTREO DE PARTÍCULAS ===
        # Eliminar partículas con peso bajo y duplicar las de peso alto
        self.particles = self.resample_particles(self.particles)

        # === 4. ESTIMACIÓN DE POSE USANDO PROMEDIO PONDERADO ===
        # La pose estimada del robot es el promedio ponderado de todas las partículas
        weighted_x = sum(p.x * p.weight for p in self.particles)
        weighted_y = sum(p.y * p.weight for p in self.particles)
        weighted_theta = sum(p.theta * p.weight for p in self.particles)
        self.current_map_pose = [weighted_x, weighted_y, weighted_theta]

        # === 5. ACTUALIZACIÓN DEL MAPA ===
        # Cada partícula actualiza su propio mapa con el scan actual
        for p in self.particles:
            self.update_map(p, msg)

        # === 6. PUBLICAR TRANSFORMACIÓN map->odom ===
        # Mantener consistencia en el árbol de transformaciones TF
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        """
        Calcula el peso de una partícula basado en qué tan bien coincide 
        el scan láser actual con el mapa de esa partícula
        
        Método simple pero efectivo: cuenta cuántos endpoints del láser
        coinciden con celdas ocupadas en el mapa de la partícula.
        
        Args:
            particle: Partícula a evaluar
            scan_msg: Mensaje de LaserScan actual
            
        Returns:
            float: Peso de la partícula (mayor = mejor coincidencia)
        """
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        
        hit_count = 0           # Contador de coincidencias
        total_valid_readings = 0  # Total de lecturas válidas
        
        # Evaluar cada rayo del laser scan
        for i, range_dist in enumerate(scan_msg.ranges):
            # Filtrar lecturas válidas (no NaN, dentro del rango)
            if (math.isnan(range_dist) or 
                range_dist < scan_msg.range_min or 
                range_dist >= scan_msg.range_max):
                continue
            
            # Calcular endpoint del rayo láser en coordenadas del mundo
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            endpoint_x = robot_x + range_dist * math.cos(robot_theta + angle)
            endpoint_y = robot_y + range_dist * math.sin(robot_theta + angle)
            
            # Convertir a coordenadas del mapa (píxeles)
            map_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            map_y = int((endpoint_y - self.map_origin_y) / self.resolution)
            
            # Verificar si está dentro del mapa
            if (0 <= map_x < self.map_width_cells and 
                0 <= map_y < self.map_height_cells):
                
                total_valid_readings += 1
                
                # Si la celda está ocupada (log_odds > 0), es un hit
                if particle.log_odds_map[map_y, map_x] > 0:
                    hit_count += 1
        
        # Evitar división por cero
        if total_valid_readings == 0:
            return 0.1  # Peso mínimo
        
        # Peso = proporción de hits + un mínimo para evitar peso cero
        weight = (hit_count / total_valid_readings) + 0.1
        
        return weight

    def resample_particles(self, particles):
        """
        Remuestrea las partículas usando Stochastic Universal Sampling (SUS)
        
        Este proceso elimina partículas con peso bajo y duplica las de peso alto,
        manteniendo la diversidad del filtro de partículas.
        
        Args:
            particles: Lista de partículas actuales
            
        Returns:
            list: Nueva lista de partículas remuestreadas
        """
        weights = [p.weight for p in particles]
        total_weight = sum(weights)
        
        # Si no hay peso válido, distribución uniforme
        if total_weight <= 0:
            for p in particles:
                p.weight = 1.0 / len(particles)
            return particles
        
        # Normalizar pesos
        weights = [w / total_weight for w in weights]
        
        # Calcular pesos acumulativos
        cumulative = np.cumsum(weights)
        
        # SUS: una sola muestra aleatoria, luego espaciado uniforme
        step = 1.0 / len(particles)
        start = np.random.uniform(0, step)
        pointers = [start + i * step for i in range(len(particles))]
        
        # Remuestreo
        new_particles = []
        for pointer in pointers:
            # Encontrar índice donde cumulative >= pointer
            idx = np.searchsorted(cumulative, pointer)
            if idx >= len(particles):
                idx = len(particles) - 1
            
            # Copiar partícula seleccionada (clonar completamente)
            p = particles[idx]
            new_particle = Particle(p.x, p.y, p.theta, 1.0/len(particles), 
                                  (self.map_height_cells, self.map_width_cells))
            new_particle.log_odds_map = p.log_odds_map.copy()  # Copiar el mapa
            new_particle.prev_odom_x = p.prev_odom_x
            new_particle.prev_odom_y = p.prev_odom_y
            new_particle.prev_odom_theta = p.prev_odom_theta
            new_particles.append(new_particle)
        
        return new_particles

    def update_map(self, particle, scan_msg):
        """
        Actualiza el mapa de una partícula usando el scan láser actual
        
        Para cada rayo del láser:
        1. Marca celdas libres a lo largo del rayo (hasta el obstáculo)
        2. Marca celda ocupada en el endpoint (si hay obstáculo)
        
        Args:
            particle: Partícula cuyo mapa se va a actualizar
            scan_msg: Mensaje de LaserScan actual
        """
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        
        # Procesar cada rayo del laser scan
        for i, range_dist in enumerate(scan_msg.ranges):
            # Determinar si el rayo golpeó un obstáculo
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            
            # Filtrar lecturas inválidas
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue
            
            # Calcular ángulo del rayo y endpoint
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            endpoint_x = robot_x + current_range * math.cos(robot_theta + angle)
            endpoint_y = robot_y + current_range * math.sin(robot_theta + angle)
            
            # Convertir a coordenadas del mapa
            robot_map_x = int((robot_x - self.map_origin_x) / self.resolution)
            robot_map_y = int((robot_y - self.map_origin_y) / self.resolution)
            endpoint_map_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            endpoint_map_y = int((endpoint_y - self.map_origin_y) / self.resolution)

            # Marcar celdas libres a lo largo del rayo usando algoritmo de Bresenham
            self.bresenham_line(particle, robot_map_x, robot_map_y, endpoint_map_x, endpoint_map_y)

            # Marcar endpoint como ocupado si es un hit válido
            if is_hit and 0 <= endpoint_map_x < self.map_width_cells and 0 <= endpoint_map_y < self.map_height_cells:
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] += self.log_odds_occupied
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] = np.clip(
                    particle.log_odds_map[endpoint_map_y, endpoint_map_x], 
                    self.log_odds_min, self.log_odds_max)

    def bresenham_line(self, particle, x0, y0, x1, y1):
        """
        Implementa el algoritmo de línea de Bresenham para marcar celdas libres
        a lo largo de un rayo láser
        
        Este algoritmo eficientemente determina qué celdas del mapa están
        en la línea recta entre el robot y el endpoint del rayo.
        
        Args:
            particle: Partícula cuyo mapa se actualiza
            x0, y0: Coordenadas inicio (robot) en celdas del mapa  
            x1, y1: Coordenadas fin (endpoint) en celdas del mapa
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy  # Prevenir bucles infinitos
        
        # Recorrer la línea marcando celdas como libres
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            # Verificar que estemos dentro del mapa
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                # Actualizar log-odds hacia "libre"
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(
                    particle.log_odds_map[y0, x0], 
                    self.log_odds_min, self.log_odds_max)
            
            # Algoritmo de Bresenham
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        """
        Publica el mapa actual como OccupancyGrid para visualización en RViz
        
        Usa el mapa de la partícula con mayor peso como mapa representativo.
        Convierte log-odds a valores de ocupación estándar:
        - 0 = libre
        - 100 = ocupado  
        - -1 = desconocido
        """
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value

        # Configurar metadatos del mapa
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Usar la partícula con mayor peso para el mapa publicado
        best_particle = max(self.particles, key=lambda p: p.weight)

        # Convertir log-odds a occupancy grid usando NumPy
        log_odds_map = best_particle.log_odds_map
        occupancy_grid = np.full_like(log_odds_map, -1, dtype=np.int8)  # Inicializar como desconocido
        occupancy_grid[log_odds_map > 0.5] = 100    # Occupied
        occupancy_grid[log_odds_map < -0.5] = 0     # Free

        # Convertir a lista y publicar
        map_msg.data = occupancy_grid.flatten().tolist()
        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Mapa publicado correctamente.")

    def get_odom_transform(self):
        """
        Obtiene la transformación actual de odometría (odom → base_link) del árbol TF
        
        Esta transformación es necesaria para calcular la transformación map → odom
        que mantiene la consistencia del árbol de transformaciones.
        
        Returns:
            TransformStamped: Transformación odom → base_link, o None si falla
        """
        try:
            return self.tf_buffer.lookup_transform(
                self.get_parameter('odom_frame').get_parameter_value().string_value, 
                self.get_parameter('base_frame').get_parameter_value().string_value, 
                rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f'No se pudo obtener transformación odom→base_link: {e}', throttle_duration_sec=2.0)
            return None

    def pose_to_matrix(self, pose):
        """
        Convierte una pose (posición + orientación) a matriz de transformación homogénea 4x4
        
        Las matrices homogéneas permiten combinar rotaciones y traslaciones
        en una sola operación matemática.
        
        Args:
            pose: Pose con position y orientation
            
        Returns:
            numpy.ndarray: Matriz 4x4 de transformación homogénea
        """
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return mat

    def transform_to_matrix(self, transform):
        """
        Convierte una transformación TF a matriz homogénea 4x4
        
        Args:
            transform: Transform con translation y rotation
            
        Returns:
            numpy.ndarray: Matriz 4x4 de transformación homogénea
        """
        q = transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        t = transform.translation
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def broadcast_map_to_odom(self):
        """
        Publica la transformación map → odom necesaria para el árbol de transformaciones TF
        
        Esta transformación cierra el bucle del árbol TF:
        map → odom → base_link
        
        Cálculo: map → odom = map → base_link × inv(odom → base_link)
        
        Donde:
        - map → base_link es nuestra estimación SLAM
        - odom → base_link viene de la odometría del robot
        """
        # Obtener transformación actual de odometría
        odom_tf = self.get_odom_transform()
        if odom_tf is None:
            self.get_logger().warn("No se puede publicar transformación map→odom: falta transformación odom→base_link")
            return
        
        # Crear pose estimada del robot en el frame del mapa
        estimated_pose = Pose()
        estimated_pose.position.x = self.current_map_pose[0]
        estimated_pose.position.y = self.current_map_pose[1] 
        estimated_pose.position.z = 0.0
        
        # Convertir ángulo yaw a cuaternión
        quaternion = R.from_euler('z', self.current_map_pose[2]).as_quat()
        estimated_pose.orientation = Quaternion(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )
        
        # Calcular transformación map → odom usando matrices homogéneas
        map_to_base_matrix = self.pose_to_matrix(estimated_pose)      # map → base_link (estimado)
        odom_to_base_matrix = self.transform_to_matrix(odom_tf.transform)  # odom → base_link (odometría)
        
        # Aplicar relación: map_to_odom = map_to_base × inv(odom_to_base)
        map_to_odom_matrix = np.dot(map_to_base_matrix, np.linalg.inv(odom_to_base_matrix))
        
        # Construir mensaje de transformación
        transform_msg = TransformStamped()
        
        # Extraer componentes de traslación y rotación de la matriz 4x4
        translation = map_to_odom_matrix[:3, 3]
        rotation_quat = R.from_matrix(map_to_odom_matrix[:3, :3]).as_quat()

        # Llenar datos del mensaje
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        transform_msg.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
        transform_msg.transform.translation.x = translation[0]
        transform_msg.transform.translation.y = translation[1]
        transform_msg.transform.translation.z = translation[2]
        transform_msg.transform.rotation = Quaternion(
            x=rotation_quat[0], y=rotation_quat[1], z=rotation_quat[2], w=rotation_quat[3]
        )

        # Publicar transformación al árbol TF
        self.tf_broadcaster.sendTransform(transform_msg)

    @staticmethod
    def angle_diff(a, b):
        """
        Calcula la diferencia angular más corta entre dos ángulos de forma robusta
        
        Esta función es crucial para evitar problemas cuando los ángulos cruzan
        la discontinuidad de -π a π.
        
        Args:
            a, b: ángulos en radianes
        Returns:
            float: diferencia angular en [-π, π]
        """
        # Verificar valores válidos
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            return 0.0
        
        # Usar normalize_angle para mayor robustez numérica
        diff = normalize_angle(a - b)
        return diff

def main(args=None):
    """
    Función principal que inicializa y ejecuta el nodo de SLAM
    
    Args:
        args: Argumentos de línea de comandos
    """
    rclpy.init(args=args)
    node = PythonSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()