import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import grey_dilation
from scipy.ndimage import distance_transform_edt
import heapq
from enum import Enum

from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseArray, TransformStamped, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformListener, Buffer

import math
from geometry_msgs.msg import Twist, Vector3

class State(Enum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    AVOIDING_OBSTACLE = 3

class AmclNode(Node):
    def __init__(self):
        super().__init__('my_py_amcl')

        # === Declaración de parámetros de configuración del sistema ===
        # Parámetros de frames y tópicos ROS
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('initial_pose_topic', '/initialpose')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        
        # Parámetros de configuración del sensor láser
        self.declare_parameter('laser_max_range', 3.5)  # Alcance máximo efectivo del sensor láser en metros
        
        # Parámetros de navegación y evasión de obstáculos
        self.declare_parameter('obstacle_detection_distance', 0.3)  # Distancia mínima para considerar un obstáculo (metros)
        self.declare_parameter('obstacle_avoidance_turn_speed', 0.5)  # Velocidad angular durante maniobras de evasión (rad/s)
        self.declare_parameter('angular_gain', 1.5)  # Ganancia proporcional para el control angular en navegación
        self.declare_parameter('max_curvature', 3.0)  # Curvatura máxima permitida para limitar giros bruscos (rad/m)

        # === Parámetros del filtro de partículas AMCL ===
        self.declare_parameter('num_particles', 100)  # Número total de partículas para representar la distribución de pose
        
        # Parámetros de ruido del modelo de movimiento (motion model noise parameters)
        self.declare_parameter('alpha1', 0.005)  # Ruido rotacional proporcional a la rotación (rad²/rad²)
        self.declare_parameter('alpha2', 0.005)  # Ruido rotacional proporcional a la traslación (rad²/m²)
        self.declare_parameter('alpha3', 0.01)   # Ruido traslacional proporcional a la traslación (m²/m²)
        self.declare_parameter('alpha4', 0.01)   # Ruido traslacional proporcional a la rotación (m²/rad²)
        
        # Parámetros del modelo de observación del sensor láser
        self.declare_parameter('z_hit', 0.9)   # Probabilidad de que una lectura del láser coincida con el mapa
        self.declare_parameter('z_rand', 0.1)  # Probabilidad de lecturas aleatorias debido a ruido del sensor
        
        # Parámetros de navegación autónoma con Pure Pursuit
        self.declare_parameter('lookahead_distance', 0.4)     # Distancia de anticipación para el algoritmo Pure Pursuit (metros)
        self.declare_parameter('linear_velocity', 0.1)        # Velocidad lineal base para navegación autónoma (m/s)
        self.declare_parameter('goal_tolerance', 0.10)        # Radio de tolerancia para considerar alcanzado el objetivo (metros)
        self.declare_parameter('angular_tolerance', 0.1)      # Tolerancia angular para considerar alcanzada la orientación objetivo (radianes)
        self.declare_parameter('final_rotation_speed', 0.3)   # Velocidad angular para rotación final hacia orientación objetivo (rad/s)
        self.declare_parameter('path_pruning_distance', 0.3)  # Distancia para eliminar waypoints ya visitados (metros)
        
        # Parámetros de seguridad y inflado de mapa
        self.declare_parameter('safety_margin_cells', 4)  # Número de celdas a expandir alrededor de obstáculos como margen de seguridad
        
        # Parámetros de ruido para el remuestreo de partículas
        self.declare_parameter('resample_noise_x', 0.05)    # Desviación estándar del ruido en coordenada X durante remuestreo (metros)
        self.declare_parameter('resample_noise_y', 0.05)    # Desviación estándar del ruido en coordenada Y durante remuestreo (metros)
        self.declare_parameter('resample_noise_yaw', 0.02)  # Desviación estándar del ruido angular durante remuestreo (radianes)

        
        # === Inicialización de variables internas a partir de parámetros ===
        self.num_particles = self.get_parameter('num_particles').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.laser_max_range = self.get_parameter('laser_max_range').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        
        # Vector de parámetros de ruido del modelo de movimiento según Thrun et al.
        self.alphas = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])
        
        # Parámetros de control de navegación
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.angular_tolerance = self.get_parameter('angular_tolerance').value
        self.final_rotation_speed = self.get_parameter('final_rotation_speed').value
        self.path_pruning_distance = self.get_parameter('path_pruning_distance').value
        self.safety_margin_cells = self.get_parameter('safety_margin_cells').value
        self.obstacle_detection_distance = self.get_parameter('obstacle_detection_distance').value
        self.obstacle_avoidance_turn_speed = self.get_parameter('obstacle_avoidance_turn_speed').value
        self.max_curvature = self.get_parameter('max_curvature').value
        self.angular_gain = self.get_parameter('angular_gain').value
        
        # Parámetros de ruido para diversificar partículas durante remuestreo
        self.resample_noise = [
            self.get_parameter('resample_noise_x').value,
            self.get_parameter('resample_noise_y').value,
            self.get_parameter('resample_noise_yaw').value
        ]

        # === Variables de estado del filtro de partículas ===
        self.particles = np.zeros((self.num_particles, 3))  # Array de partículas [x, y, yaw]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Pesos normalizados de cada partícula
        
        # === Variables de estado del sistema ===
        self.map_data = None  # Datos del mapa de ocupación recibido
        self.latest_scan = None  # Última lectura del sensor láser
        self.initial_pose_received = False  # Flag para indicar si se recibió pose inicial
        self.map_received = False  # Flag para indicar si se recibió el mapa
        self.last_odom_pose = None  # Última pose de odometría para calcular incrementos
        
        # === Variables de navegación autónoma ===
        self.state = State.IDLE  # Estado actual de la máquina de estados de navegación
        self.current_path = None  # Camino actual planificado hacia el objetivo
        self.goal_pose = None  # Pose objetivo actual
        self.inflated_grid = None  # Mapa inflado con márgenes de seguridad
        self.rotating_to_final_orientation = False  # Flag para indicar si está rotando hacia orientación final
        
        # === Variables para evasión de obstáculos ===
        self.obstacle_avoidance_start_yaw = None  # Orientación inicial al comenzar evasión
        self.obstacle_avoidance_last_yaw = None  # Última orientación durante evasión
        self.obstacle_avoidance_cumulative_angle = 0.0  # Ángulo total girado durante evasión
        self.obstacle_avoidance_active = False  # Flag para indicar si está evadiendo obstáculos
        
        # === Configuración de interfaces ROS 2 ===
        # Perfiles de calidad de servicio (QoS) optimizados para diferentes tipos de datos
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,      # Garantiza entrega del mapa
            history=QoSHistoryPolicy.KEEP_LAST,             # Solo mantiene el último mapa
            depth=1,                                         # Buffer de tamaño 1
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL  # Persiste para suscriptores tardíos
        )
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,   # Prioriza velocidad sobre garantías
            history=QoSHistoryPolicy.KEEP_LAST,             # Solo mantiene scans recientes
            depth=10                                         # Buffer para múltiples scans
        )
        
        # === Suscriptores ===
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, scan_qos)
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, self.get_parameter('initial_pose_topic').value, self.initial_pose_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)
        
        # === Publicadores ===
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 10)  # Pose estimada del robot
        self.particle_pub = self.create_publisher(MarkerArray, 'particle_cloud', 10)  # Visualización de partículas en RViz
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)  # Comandos de velocidad
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)  # Camino planificado para visualización
        
        # === Manejo de transformaciones ===
        self.tf_buffer = Buffer()  # Buffer para almacenar transformaciones recientes
        self.tf_listener = TransformListener(self.tf_buffer, self)  # Escucha transformaciones del árbol TF
        self.tf_broadcaster = TransformBroadcaster(self)  # Publica transformaciones al árbol TF

        # === Timer principal del sistema ===
        self.timer = self.create_timer(0.1, self.timer_callback)  # Ejecuta ciclo principal a 10 Hz
        self.get_logger().info('Nodo de localización AMCL inicializado correctamente')

    def map_callback(self, msg):
        """
        Callback para procesar el mapa de ocupación recibido.
        Realiza el inflado del mapa y precalcula el campo de distancias para localización.
        """
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            
            # Convertir datos del mapa a matriz 2D numpy para procesamiento eficiente
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            
            # Inflar obstáculos con margen de seguridad para navegación
            self.inflate_map()
            
            # Precalcular campo de distancias para modelo de observación eficiente
            # Las celdas ocupadas tienen valores > 50 según convención de OccupancyGrid
            occupied = self.grid > 50
            # Transformada de distancia euclidiana: distancia a la celda ocupada más cercana
            self.dist_field = distance_transform_edt(~occupied).astype(np.float32) * self.map_data.info.resolution
            
            self.get_logger().info('Mapa recibido y procesado correctamente')


    def scan_callback(self, msg):
        """
        Callback para almacenar la última lectura del sensor láser.
        """
        self.latest_scan = msg

    def goal_callback(self, msg):
        """
        Callback para procesar nuevos objetivos de navegación.
        Valida el frame del objetivo y activa el estado de planificación.
        """
        if self.map_data is None:
            self.get_logger().warn("Objetivo recibido pero el mapa aún no está disponible. Ignorando objetivo.")
            return

        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Objetivo recibido en frame '{msg.header.frame_id}', pero se esperaba '{self.map_frame_id}'. Ignorando.")
            return
            
        self.goal_pose = msg.pose
        self.get_logger().info(f"Nuevo objetivo: ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f})")
        self.state = State.PLANNING
        self.current_path = None
        self.rotating_to_final_orientation = False  # Reiniciar flag de rotación

    def initial_pose_callback(self, msg):
        """
        Callback para procesar la pose inicial del robot.
        Inicializa el filtro de partículas y publica las transformaciones correspondientes.
        """
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Frame de pose inicial '{msg.header.frame_id}' no coincide con frame del mapa '{self.map_frame_id}'. Ignorando.")
            return
            
        self.get_logger().info('Pose inicial recibida - Inicializando filtro de partículas')
        
        # Inicializar partículas alrededor de la pose recibida
        self.initialize_particles(msg.pose.pose)
        
        # Publicar pose estimada inicial
        self.publish_pose(msg.pose.pose)
        
        # Obtener y publicar transformación inicial map->odom
        odom_tf = self.get_odom_transform()
        if odom_tf is not None:
            self.publish_transform(msg.pose.pose, odom_tf)
        else:
            self.get_logger().warn("No se pudo obtener transformación odom→base para inicialización")
        
        # Activar sistema de localización
        self.state = State.IDLE
        self.initial_pose_received = True
        self.last_odom_pose = None  # Reiniciar seguimiento de odometría
        self.rotating_to_final_orientation = False  # Reiniciar flag de rotación
        self.stop_robot()



    def initialize_particles(self, initial_pose):
        """
        Inicializa las partículas del filtro alrededor de una pose inicial conocida.
        Distribuye las partículas con ruido gaussiano para representar incertidumbre inicial.
        
        Args:
            initial_pose: Pose inicial del robot como punto de referencia
        """
        # Extraer coordenadas del centro de distribución
        mean_x = initial_pose.position.x
        mean_y = initial_pose.position.y

        # Convertir orientación de cuaternión a ángulo yaw
        q = initial_pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]

        # Desviaciones estándar para la distribución inicial de incertidumbre
        std_position = 0.5   # 50 cm de incertidumbre en posición
        std_orientation = 0.3  # 0.3 rad (~17°) de incertidumbre en orientación

        # Generar distribución gaussiana de partículas alrededor de la pose inicial
        self.particles[:, 0] = np.random.normal(mean_x, std_position, self.num_particles)     # coordenada x
        self.particles[:, 1] = np.random.normal(mean_y, std_position, self.num_particles)     # coordenada y
        self.particles[:, 2] = np.random.normal(yaw, std_orientation, self.num_particles)     # orientación yaw

        # Normalizar ángulos al rango [-π, π]
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # Inicializar todos los pesos como equiprobables
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Actualizar visualización en RViz
        self.publish_particles()


    def initialize_particles_randomly(self):
        """
        Inicializa partículas distribuyéndolas aleatoriamente por todo el espacio libre del mapa.
        Se utiliza cuando no se conoce la pose inicial del robot (localización global).
        """
        self.get_logger().info("Inicializando partículas aleatoriamente en espacio libre del mapa")
        
        # Encontrar todas las celdas libres (valor 0 en OccupancyGrid)
        free_indices = np.where(np.array(self.map_data.data) == 0)[0]

        if len(free_indices) == 0:
            self.get_logger().warn("No existen celdas libres en el mapa para colocar partículas")
            return

        # Seleccionar aleatoriamente posiciones libres para las partículas
        chosen_indices = np.random.choice(free_indices, self.num_particles)
        map_w = self.map_data.info.width

        # Convertir índices lineales a coordenadas de grilla y luego a coordenadas del mundo
        xs = []
        ys = []
        for idx in chosen_indices:
            gy = idx // map_w  # fila en la grilla
            gx = idx % map_w   # columna en la grilla
            wx, wy = self.grid_to_world(gx, gy)
            xs.append(wx)
            ys.append(wy)

        # Asignar posiciones calculadas y orientaciones aleatorias
        self.particles[:, 0] = xs  # coordenadas x en el mundo
        self.particles[:, 1] = ys  # coordenadas y en el mundo
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)  # orientaciones aleatorias
        
        # Inicializar pesos equiprobables
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Actualizar visualización
        self.publish_particles()


    def timer_callback(self):
        """
        Función principal del ciclo de control que se ejecuta a 10 Hz.
        Maneja la localización, navegación y máquina de estados del robot.
        """
        # === Verificar precondiciones básicas del sistema ===
        if not self.map_received or self.latest_scan is None:
            return
        
        # === Inicialización automática de partículas para localización global ===
        if not self.initial_pose_received:
            # Si no se ha recibido pose inicial, inicializar partículas aleatoriamente
            # para hacer localización global (Global Localization)
            self.initialize_particles_randomly()
            self.initial_pose_received = True  # Marcar como inicializado
            self.get_logger().info("Iniciando localización global - partículas distribuidas aleatoriamente")
            return

        # === Obtener transformación actual de odometría ===
        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            self.stop_robot()
            return

        # === Ejecutar ciclo de localización AMCL ===
        estimated_pose = self.run_localization_cycle(current_odom_tf)

        # === Publicar estimaciones de estado ===
        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)

        # === Máquina de estados de navegación ===
        if self.state == State.IDLE:
            # Permanecer detenido hasta recibir un nuevo objetivo
            self.stop_robot()
            return

        elif self.state == State.PLANNING:
            # Ejecutar planificación A* y transición a NAVIGATING
            self.handle_planning_state(estimated_pose)
            return

        elif self.state == State.NAVIGATING:
            # Seguir trayectoria con Pure Pursuit; puede transicionar a IDLE o AVOIDING_OBSTACLE
            self.handle_navigating_state(estimated_pose)
            return

        elif self.state == State.AVOIDING_OBSTACLE:
            # Ejecutar maniobra de evasión de obstáculos; regresa a NAVIGATING
            self.handle_avoiding_obstacle_state(estimated_pose)
            return



    def get_odom_transform(self):
        try:
            return self.tf_buffer.lookup_transform(self.odom_frame_id, self.base_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Skipping update. Error: {e}', throttle_duration_sec=2.0)
            return None


    def motion_model(self, current_odom_tf):
        """
        Modelo de movimiento probabilístico basado en odometría.
        Implementa el algoritmo 'sample_motion_model_odometry' de Thrun et al.
        
        Características:
        - Distingue entre rotación in-situ y desplazamiento
        - Calcula varianzas según ecuaciones de Probabilistic Robotics
        - Maneja correctamente la normalización angular
        
        Args:
            current_odom_tf: Transformación actual de odometría
        """
        # En la primera ejecución, solo inicializar referencia temporal
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_tf.transform
            return

        # === Calcular incremento de traslación ===
        t1 = self.last_odom_pose.translation
        t2 = current_odom_tf.transform.translation
        dx, dy = t2.x - t1.x, t2.y - t1.y
        delta_trans = np.hypot(dx, dy)

        # === Calcular incrementos de rotación ===
        q1 = self.last_odom_pose.rotation
        q2 = current_odom_tf.transform.rotation
        theta1 = R.from_quat([q1.x, q1.y, q1.z, q1.w]).as_euler('zyx')[0]
        theta2 = R.from_quat([q2.x, q2.y, q2.z, q2.w]).as_euler('zyx')[0]

        def normalize_angle(angle):
            """Normaliza ángulo al rango [-π, π]"""
            return (angle + np.pi) % (2*np.pi) - np.pi

        # Descomponer movimiento en rotación inicial, traslación y rotación final
        if delta_trans < 1e-4:
            # Caso especial: rotación pura sin desplazamiento
            delta_rot1 = 0.0
            delta_rot2 = normalize_angle(theta2 - theta1)
        else:
            # Caso general: movimiento con traslación
            delta_rot1 = normalize_angle(np.arctan2(dy, dx) - theta1)  # rotación hacia dirección de movimiento
            delta_rot2 = normalize_angle((theta2 - theta1) - delta_rot1)  # rotación final

        # Omitir actualización si no hubo movimiento significativo
        if delta_trans < 1e-4 and abs(normalize_angle(theta2 - theta1)) < 1e-4:
            self.last_odom_pose = current_odom_tf.transform
            return

        # === Calcular varianzas del ruido según modelo probabilístico ===
        a1, a2, a3, a4 = self.alphas
        var_rot1 = a1 * delta_rot1**2 + a2 * delta_trans**2
        var_trans = a3 * delta_trans**2 + a4 * (delta_rot1**2 + delta_rot2**2)
        var_rot2 = a1 * delta_rot2**2 + a2 * delta_trans**2

        # === Generar muestras ruidosas del movimiento (vectorizado) ===
        delta_rot1_noisy = delta_rot1 + np.random.normal(0, np.sqrt(var_rot1), self.num_particles)
        delta_trans_noisy = delta_trans + np.random.normal(0, np.sqrt(var_trans), self.num_particles)
        delta_rot2_noisy = delta_rot2 + np.random.normal(0, np.sqrt(var_rot2), self.num_particles)

        # === Aplicar movimiento ruidoso a todas las partículas ===
        self.particles[:, 0] += delta_trans_noisy * np.cos(self.particles[:, 2] + delta_rot1_noisy)
        self.particles[:, 1] += delta_trans_noisy * np.sin(self.particles[:, 2] + delta_rot1_noisy)
        self.particles[:, 2] = normalize_angle(self.particles[:, 2] + delta_rot1_noisy + delta_rot2_noisy)

        # Actualizar referencia para próxima iteración
        self.last_odom_pose = current_odom_tf.transform


    def measurement_model(self):
        """
        Modelo de observación del sensor láser para actualizar pesos de partículas.
        Utiliza el algoritmo de likelihood field para evaluar la correspondencia
        entre las lecturas del láser y el mapa conocido.
        
        Implementa un modelo de mezcla que considera:
        - Lecturas correctas (z_hit): Coincidencias con obstáculos del mapa
        - Ruido aleatorio (z_rand): Lecturas espurias del sensor
        """
        if self.latest_scan is None or self.map_data is None:
            return

        # === Parámetros del modelo de observación ===
        sigma_hit = 0.12     # Desviación estándar del sensor láser (12 cm)
        z_hit = 0.80         # Peso del modelo de coincidencia
        z_rand = 0.20        # Peso del modelo de ruido aleatorio
        max_range = self.laser_max_range

        # === Preprocesar datos del láser ===
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        angles = np.arange(
            self.latest_scan.angle_min,
            self.latest_scan.angle_max,
            self.latest_scan.angle_increment,
            dtype=np.float32
        )

        # Submuestreo para eficiencia computacional (mantener tiempo real)
        decimation_factor = 8  # Usar aproximadamente 45 rayos de 360
        ranges = ranges[::decimation_factor]
        angles = angles[::decimation_factor]

        # === Calcular pesos para cada partícula ===
        weights = np.zeros(self.num_particles, dtype=np.float32)

        for i, (px, py, ptheta) in enumerate(self.particles):
            # Precalcular funciones trigonométricas para todos los ángulos
            global_angles = ptheta + angles
            cos_angles = np.cos(global_angles)
            sin_angles = np.sin(global_angles)

            # Calcular endpoints de los rayos láser en coordenadas globales
            endpoint_x = px + ranges * cos_angles
            endpoint_y = py + ranges * sin_angles

            # Convertir a índices de celda del mapa
            grid_x = ((endpoint_x - self.map_data.info.origin.position.x) /
                     self.map_data.info.resolution).astype(np.int32)
            grid_y = ((endpoint_y - self.map_data.info.origin.position.y) /
                     self.map_data.info.resolution).astype(np.int32)

            # Filtrar puntos dentro de los límites del mapa
            valid_mask = ((grid_x >= 0) & (grid_x < self.map_data.info.width) &
                         (grid_y >= 0) & (grid_y < self.map_data.info.height) &
                         ~np.isnan(ranges))

            # Obtener distancias a obstáculos más cercanos desde campo precomputado
            distances = np.full_like(ranges, max_range, dtype=np.float32)
            valid_indices = valid_mask.nonzero()[0]
            distances[valid_indices] = self.dist_field[grid_y[valid_indices], grid_x[valid_indices]]

            # === Aplicar modelo de mezcla de probabilidades ===
            # Componente de coincidencia: distribución gaussiana centrada en obstáculos
            p_hit_component = (z_hit * np.exp(-0.5 * (distances / sigma_hit)**2) /
                              (sigma_hit * np.sqrt(2 * np.pi)))
            
            # Componente de ruido: distribución uniforme
            p_rand_component = z_rand / max_range
            
            # Probabilidad total como mezcla de componentes
            total_probability = p_hit_component + p_rand_component

            # Calcular peso final como producto de probabilidades (en log-space para estabilidad)
            log_likelihood = np.sum(np.log(total_probability + 1e-12))
            weights[i] = np.exp(log_likelihood)

        # === Normalizar pesos y evitar división por cero ===
        weights += 1e-300  # Evitar pesos exactamente cero
        self.weights = weights / np.sum(weights)


    def resample(self):
        """
        Realiza el remuestreo de partículas basado en sus pesos de importancia.
        
        Proceso:
        1. Selecciona partículas con probabilidad proporcional a sus pesos
        2. Añade ruido gaussiano para mantener diversidad
        3. Reinicia todos los pesos como equiprobables
        
        Previene el colapso del filtro manteniendo diversidad en la población.
        """
        # === Remuestreo estocástico basado en pesos ===
        selected_indices = np.random.choice(
            self.num_particles, 
            size=self.num_particles, 
            p=self.weights
        )
        self.particles = self.particles[selected_indices]
        
        # === Inyección de ruido para mantener diversidad ===
        # Evita que todas las partículas colapsen en una sola hipótesis
        noise_x = np.random.normal(0, self.resample_noise[0], self.num_particles)
        noise_y = np.random.normal(0, self.resample_noise[1], self.num_particles)
        noise_yaw = np.random.normal(0, self.resample_noise[2], self.num_particles)
        
        self.particles[:, 0] += noise_x   # Ruido en coordenada X
        self.particles[:, 1] += noise_y   # Ruido en coordenada Y  
        self.particles[:, 2] += noise_yaw # Ruido en orientación
        
        # === Normalización angular ===
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi
        
        # === Reinicializar pesos uniformemente ===
        self.weights = np.ones(self.num_particles) / self.num_particles



    def estimate_pose(self):
        """
        Estima la pose más probable del robot a partir de las partículas ponderadas.
        
        Utiliza:
        - Media ponderada para posición (x, y)
        - Promedio circular ponderado para orientación (evita discontinuidades en ±π)
        
        Returns:
            Pose: Estimación de la pose más probable del robot
        """
        # === Calcular posición estimada mediante media ponderada ===
        estimated_x = np.average(self.particles[:, 0], weights=self.weights)
        estimated_y = np.average(self.particles[:, 1], weights=self.weights)

        # === Calcular orientación estimada mediante promedio circular ===
        # El promedio circular evita problemas en la discontinuidad ±π
        sin_weighted = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_weighted = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        estimated_yaw = np.arctan2(sin_weighted, cos_weighted)

        # === Construir mensaje Pose con la estimación ===
        estimated_pose = Pose()
        estimated_pose.position.x = estimated_x
        estimated_pose.position.y = estimated_y
        estimated_pose.position.z = 0.0  # Robot opera en 2D
        
        # Convertir ángulo yaw a cuaternión para orientación 3D
        quaternion = R.from_euler('z', estimated_yaw).as_quat()
        estimated_pose.orientation = Quaternion(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )

        return estimated_pose

    def publish_pose(self, estimated_pose):
        p = PoseWithCovarianceStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = self.map_frame_id
        p.pose.pose = estimated_pose
        self.pose_pub.publish(p)

    def publish_particles(self):
        ma = MarkerArray()
        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            q = R.from_euler('z', p[2]).as_quat()
            marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 0.5
            marker.color.r = 1.0
            ma.markers.append(marker)
        self.particle_pub.publish(ma)

    def publish_transform(self, estimated_pose, odom_tf):
        """
        Publica la transformación map → odom necesaria para el árbol de transformaciones.
        
        Cálculo:
        map → odom = map → base_link × inv(odom → base_link)
        
        Esta transformación permite que el árbol TF sea consistente:
        map → odom → base_link
        
        Args:
            estimated_pose: Pose estimada del robot en el frame del mapa
            odom_tf: Transformación actual de odometría
        """
        # === Calcular transformación map → odom ===
        map_to_base_matrix = self.pose_to_matrix(estimated_pose)      # map → base_link (estimado)
        odom_to_base_matrix = self.transform_to_matrix(odom_tf.transform)  # odom → base_link (odometría)
        
        # Aplicar relación: map_to_odom = map_to_base × inv(odom_to_base)
        map_to_odom_matrix = np.dot(map_to_base_matrix, np.linalg.inv(odom_to_base_matrix))
        
        # === Construir mensaje de transformación ===
        transform_msg = TransformStamped()
        
        # Extraer componentes de traslación y rotación de la matriz 4x4
        translation = map_to_odom_matrix[:3, 3]
        rotation_quat = R.from_matrix(map_to_odom_matrix[:3, :3]).as_quat()

        # Llenar datos del mensaje
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = self.map_frame_id
        transform_msg.child_frame_id = self.odom_frame_id
        transform_msg.transform.translation.x = translation[0]
        transform_msg.transform.translation.y = translation[1]
        transform_msg.transform.translation.z = translation[2]
        transform_msg.transform.rotation = Quaternion(
            x=rotation_quat[0], y=rotation_quat[1], z=rotation_quat[2], w=rotation_quat[3]
        )

        # === Publicar transformación al árbol TF ===
        self.tf_broadcaster.sendTransform(transform_msg)

    def pose_to_matrix(self, pose):
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return mat

    def transform_to_matrix(self, transform):
        q = transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        t = transform.translation
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        gy = int((wy - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.map_data.info.resolution + self.map_data.info.origin.position.x
        wy = gy * self.map_data.info.resolution + self.map_data.info.origin.position.y
        return (wx, wy)
    

    def publish_path(self, path_msg):
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame_id
        self.path_pub.publish(path_msg)

    # ======================================================================
    # FUNCIONES AUXILIARES Y DE UTILIDAD
    # ======================================================================

    def stop_robot(self):
        """
        Detiene completamente el robot enviando comandos de velocidad cero.
        """
        stop_command = Twist()
        stop_command.linear.x = 0.0
        stop_command.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_command)

    def get_yaw_from_pose(self, pose):
        """
        Extrae el ángulo yaw (rotación en Z) de una pose con orientación en cuaternión.
        
        Args:
            pose: Objeto Pose con orientación en cuaternión
            
        Returns:
            float: Ángulo yaw en radianes [-π, π]
        """
        quaternion = pose.orientation
        yaw_angle = R.from_quat([quaternion.x, quaternion.y, quaternion.z, quaternion.w]).as_euler('zyx')[0]
        return yaw_angle

    def angle_diff(self, angle_a, angle_b):
        """
        Calcula la diferencia angular más corta entre dos ángulos.
        Maneja correctamente la discontinuidad en ±π.
        
        Args:
            angle_a, angle_b: Ángulos en radianes
            
        Returns:
            float: Diferencia angular en rango [-π, π]
        """
        difference = angle_a - angle_b
        return (difference + np.pi) % (2 * np.pi) - np.pi

    def is_obstacle_detected(self):
        """
        Detecta si hay obstáculos en el sector frontal del robot.
        
        Analiza las lecturas del láser en un cono frontal de ±30° para determinar
        si existe algún obstáculo dentro de la distancia de seguridad.
        
        Returns:
            bool: True si se detecta un obstáculo peligroso, False en caso contrario
        """
        if self.latest_scan is None:
            return False

        # === Configurar sector de detección frontal ===
        front_sector_angle = math.pi / 6  # ±30 grados desde el frente del robot
        
        # === Calcular índices de rayos en el sector frontal ===
        num_rays = len(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        
        # Índice del rayo central (ángulo ≈ 0, hacia adelante)
        center_ray_index = int(-angle_min / angle_increment)
        
        # Rango de índices para el sector frontal
        front_sector_span = int(front_sector_angle / angle_increment)
        start_index = max(0, center_ray_index - front_sector_span)
        end_index = min(num_rays, center_ray_index + front_sector_span)
        
        # === Buscar obstáculos en el sector frontal ===
        for ray_index in range(start_index, end_index):
            distance = self.latest_scan.ranges[ray_index]
            # Verificar si hay un obstáculo dentro de la distancia de seguridad
            if not np.isnan(distance) and distance < self.obstacle_detection_distance:
                return True
        
        return False
    
    def astar(self, start, goal):
        """
        Implementa el algoritmo A* para planificación de rutas en grilla.
        
        Encuentra el camino óptimo entre start y goal evitando obstáculos
        del mapa inflado. Utiliza heurística de distancia Manhattan.
        
        Args:
            start: Tupla (gx, gy) con coordenadas de celda inicial
            goal: Tupla (gx, gy) con coordenadas de celda objetivo
            
        Returns:
            list: Secuencia de celdas (gx, gy) del camino, o None si no existe
        """
        def manhattan_distance(cell_a, cell_b):
            """Heurística de distancia Manhattan para A*"""
            return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])

        # === Inicializar estructuras de datos de A* ===
        # Cola de prioridad: (f_score, g_score, posición_actual, padre)
        open_set = [(manhattan_distance(start, goal), 0, start, None)]
        came_from = {}  # Diccionario para reconstruir el camino
        g_score = {start: 0}  # Costo real desde el inicio

        # === Bucle principal de A* ===
        while open_set:
            f_cost, g_cost, current_cell, parent_cell = heapq.heappop(open_set)
            
            # Evitar reexplorar nodos ya procesados
            if current_cell in came_from:
                continue
                
            # Registrar cómo llegamos a este nodo
            came_from[current_cell] = parent_cell
            
            # === Verificar si alcanzamos el objetivo ===
            if current_cell == goal:
                # Reconstruir camino desde el objetivo hasta el inicio
                path = []
                node = current_cell
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                return list(reversed(path))

            # === Explorar celdas vecinas (conectividad 4-direccional) ===
            for delta_x, delta_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor_x = current_cell[0] + delta_x
                neighbor_y = current_cell[1] + delta_y
                neighbor_cell = (neighbor_x, neighbor_y)
                
                # Verificar límites del mapa y que la celda esté libre
                if (0 <= neighbor_x < self.map_data.info.width and
                    0 <= neighbor_y < self.map_data.info.height and
                    self.inflated_grid[neighbor_y, neighbor_x] == 0):
                    
                    tentative_g_score = g_cost + 1  # Costo unitario entre celdas adyacentes
                    
                    # Si encontramos un camino mejor a esta celda
                    if (neighbor_cell not in g_score or 
                        tentative_g_score < g_score[neighbor_cell]):
                        
                        g_score[neighbor_cell] = tentative_g_score
                        f_score = tentative_g_score + manhattan_distance(neighbor_cell, goal)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor_cell, current_cell))
        
        # No se encontró camino al objetivo
        return None

    def plan_path(self, start_position, goal_position):
        """
        Planifica una ruta libre de obstáculos desde la posición actual hasta el objetivo.
        
        Utiliza el algoritmo A* sobre el mapa inflado para garantizar margen de seguridad.
        Convierte las coordenadas del mundo a grilla, ejecuta A*, y convierte de vuelta.
        
        Args:
            start_position: Objeto Position con coordenadas de inicio
            goal_position: Objeto Position con coordenadas del objetivo
            
        Returns:
            Path: Mensaje ROS con secuencia de waypoints, o None si no hay ruta
        """
        # === Convertir coordenadas del mundo a índices de grilla ===
        start_gx, start_gy = self.world_to_grid(start_position.x, start_position.y)
        goal_gx, goal_gy = self.world_to_grid(goal_position.x, goal_position.y)

        # === Validar que las coordenadas estén dentro de los límites del mapa ===
        map_width = self.map_data.info.width
        map_height = self.map_data.info.height
        
        if not (0 <= start_gx < map_width and 0 <= start_gy < map_height):
            self.get_logger().warn(f"Posición inicial fuera del mapa: grilla ({start_gx}, {start_gy})")
            return None
            
        if not (0 <= goal_gx < map_width and 0 <= goal_gy < map_height):
            self.get_logger().warn(f"Objetivo fuera del mapa: grilla ({goal_gx}, {goal_gy})")
            return None

        # === Ejecutar planificación A* en el mapa de grilla ===
        path_cells = self.astar((start_gx, start_gy), (goal_gx, goal_gy))
        if not path_cells:
            self.get_logger().warn("No se encontró ruta válida hacia el objetivo")
            return None

        # === Convertir secuencia de celdas a mensaje Path de ROS ===
        path_msg = Path()
        for grid_x, grid_y in path_cells:
            # Convertir coordenadas de grilla de vuelta al mundo
            world_x, world_y = self.grid_to_world(grid_x, grid_y)
            
            # Crear waypoint
            waypoint = PoseStamped()
            waypoint.pose.position.x = world_x
            waypoint.pose.position.y = world_y
            waypoint.pose.position.z = 0.0
            waypoint.pose.orientation.w = 1.0  # Orientación neutral
            
            path_msg.poses.append(waypoint)

        self.get_logger().info(f"Ruta planificada con {len(path_cells)} waypoints")
        return path_msg

    def find_lookahead_point(self, path, x, y):
        for pose_stamped in path.poses:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = np.hypot(px - x, py - y)
            if dist > self.lookahead_distance:
                return pose_stamped.pose
        return None  # Llegó al final


    def follow_path(self, x, y, current_pose, target_pose):
        # 1) Calcula la posición del lookahead en coords del mapa:
        dx_map = target_pose.position.x - x
        dy_map = target_pose.position.y - y
        # 2) Transforma al sistema del robot (rotación inversa de yaw):
        robot_yaw = self.get_yaw_from_pose(current_pose)
        # R^T * [dx_map, dy_map]
        x_r =  math.cos(robot_yaw)*dx_map + math.sin(robot_yaw)*dy_map
        y_r = -math.sin(robot_yaw)*dx_map + math.cos(robot_yaw)*dy_map

        # 3) Pure-Pursuit: curvatura
        L = self.lookahead_distance
        if abs(L) < 1e-6:
            kappa = 0.0
        else:
            kappa = 2.0 * y_r / (L * L)

        # 4) Genera el Twist
        twist = Twist()
        twist.linear.x  = self.linear_velocity        # v_max
        twist.angular.z = self.linear_velocity * kappa

        # 5) Saturación (por seguridad)
        twist.linear.x  = min(self.linear_velocity, twist.linear.x)
        twist.angular.z = max(min(twist.angular.z, 1.0), -1.0)

        return twist
    

    def inflate_map(self):
        """
        Infla los obstáculos del mapa añadiendo un margen de seguridad.
        
        Utiliza dilatación morfológica para expandir las zonas ocupadas,
        creando un buffer de seguridad alrededor de cada obstáculo.
        """
        # Identificar celdas ocupadas (probabilidad > 50%)
        obstacle_mask = self.grid > 50
        
        # Crear kernel circular para inflado uniforme
        kernel_size = 2 * self.safety_margin_cells + 1
        kernel = np.ones((kernel_size, kernel_size))
        
        # Aplicar dilatación morfológica para expandir obstáculos
        inflated_mask = grey_dilation(obstacle_mask.astype(np.uint8), footprint=kernel)
        
        # Convertir máscara binaria a valores de ocupación estándar
        self.inflated_grid = inflated_mask * 100

    def run_localization_cycle(self, current_odom_tf):
        """
        Ejecuta un ciclo completo del filtro de partículas AMCL.
        
        Secuencia:
        1. Modelo de movimiento: actualizar partículas según odometría
        2. Modelo de observación: calcular pesos según lecturas del láser  
        3. Remuestreo: redistribuir partículas según importancia
        4. Estimación: calcular pose más probable
        
        Args:
            current_odom_tf: Transformación actual de odometría
            
        Returns:
            Pose: Estimación actualizada de la pose del robot
        """
        self.motion_model(current_odom_tf)
        self.measurement_model()
        self.resample()
        return self.estimate_pose()
    

    def handle_planning_state(self, estimated_pose):
        """
        Maneja el estado de planificación de rutas.
        
        Ejecuta planificación A* desde la pose actual hacia el objetivo.
        Transiciones posibles:
        - PLANNING → NAVIGATING: Si se encontró una ruta válida
        - PLANNING → IDLE: Si falló la planificación o no hay objetivo
        
        Args:
            estimated_pose: Pose actual estimada del robot
        """
        # === Verificar precondiciones ===
        if self.goal_pose is None:
            self.get_logger().warn("Estado PLANNING sin objetivo válido. Transición a IDLE")
            self.state = State.IDLE
            return

        # === Ejecutar planificación de ruta ===
        planned_path = self.plan_path(estimated_pose.position, self.goal_pose.position)
        
        # === Verificar resultado de la planificación ===
        if planned_path is None or len(planned_path.poses) == 0:
            self.get_logger().warn("Falló la planificación - no existe ruta al objetivo. Transición a IDLE")
            self.state = State.IDLE
            self.stop_robot()
            return

        # === Almacenar y publicar ruta planificada ===
        self.current_path = planned_path
        self.publish_path(planned_path)

        # === Transición exitosa a navegación ===
        self.get_logger().info(f"Planificación exitosa - Iniciando navegación hacia objetivo")
        self.rotating_to_final_orientation = False  # Reiniciar flag para nueva navegación
        self.state = State.NAVIGATING



    def handle_navigating_state(self, estimated_pose):
        """
        Maneja el estado de navegación autónoma siguiendo la ruta planificada.
        
        Funcionalidades:
        - Detección de llegada al objetivo
        - Detección de obstáculos y activación de evasión
        - Poda de waypoints ya visitados
        - Control Pure Pursuit con limitación de curvatura
        
        Transiciones posibles:
        - NAVIGATING → IDLE: Al alcanzar el objetivo
        - NAVIGATING → AVOIDING_OBSTACLE: Al detectar obstáculo
        
        Args:
            estimated_pose: Pose actual estimada del robot
        """
        # === Obtener posición actual ===
        robot_x = estimated_pose.position.x
        robot_y = estimated_pose.position.y

        # === Verificar si se alcanzó el objetivo ===
        distance_to_goal = np.hypot(
            self.goal_pose.position.x - robot_x,
            self.goal_pose.position.y - robot_y
        )
        
        # Si llegó al punto pero aún no está en la orientación correcta
        if distance_to_goal < self.goal_tolerance:
            if not self.rotating_to_final_orientation:
                self.get_logger().info("Posición objetivo alcanzada - Iniciando rotación hacia orientación final")
                self.rotating_to_final_orientation = True
            
            # Calcular diferencia angular con la orientación objetivo
            current_yaw = self.get_yaw_from_pose(estimated_pose)
            goal_yaw = self.get_yaw_from_pose(self.goal_pose)
            angular_error = self.angle_diff(goal_yaw, current_yaw)
            
            # Verificar si la orientación es correcta
            if abs(angular_error) < self.angular_tolerance:
                self.get_logger().info("¡Objetivo y orientación alcanzados exitosamente!")
                self.state = State.IDLE
                self.rotating_to_final_orientation = False
                self.stop_robot()
                return
            else:
                # Rotar hacia la orientación objetivo
                rotation_command = Twist()
                rotation_command.linear.x = 0.0
                rotation_command.angular.z = (self.final_rotation_speed if angular_error > 0 
                                            else -self.final_rotation_speed)
                self.cmd_vel_pub.publish(rotation_command)
                return

        # === Detección de obstáculos en el frente ===
        # Solo detectar obstáculos si no estamos rotando hacia la orientación final
        if not self.rotating_to_final_orientation and self.is_obstacle_detected():
            self.get_logger().info("Obstáculo detectado - Iniciando maniobra de evasión")
            self.state = State.AVOIDING_OBSTACLE
            # Inicializar variables de control para evasión
            self.obstacle_avoidance_start_yaw = self.get_yaw_from_pose(estimated_pose)
            self.obstacle_avoidance_last_yaw = self.obstacle_avoidance_start_yaw
            self.obstacle_avoidance_cumulative_angle = 0.0
            return

        # === Poda de waypoints ya visitados ===
        # Solo podar waypoints si no estamos rotando hacia la orientación final
        if not self.rotating_to_final_orientation:
            # Eliminar puntos del camino que ya pasamos para optimizar búsqueda
            self.current_path.poses = [
                waypoint for waypoint in self.current_path.poses
                if np.hypot(
                    waypoint.pose.position.x - robot_x,
                    waypoint.pose.position.y - robot_y
                ) > self.path_pruning_distance
            ]

        # === Búsqueda del punto de seguimiento (lookahead) ===
        target_point = self.find_lookahead_point(self.current_path, robot_x, robot_y)
        if target_point is None:
            # Si no hay más waypoints, verificar si también se alcanzó la orientación
            if not self.rotating_to_final_orientation:
                self.get_logger().info("No hay más waypoints - Verificando orientación final")
                self.rotating_to_final_orientation = True
            
            # Calcular diferencia angular con la orientación objetivo
            current_yaw = self.get_yaw_from_pose(estimated_pose)
            goal_yaw = self.get_yaw_from_pose(self.goal_pose)
            angular_error = self.angle_diff(goal_yaw, current_yaw)
            
            # Verificar si la orientación es correcta
            if abs(angular_error) < self.angular_tolerance:
                self.get_logger().info("¡Objetivo y orientación final alcanzados exitosamente!")
                self.state = State.IDLE
                self.rotating_to_final_orientation = False
                self.stop_robot()
                return
            else:
                # Rotar hacia la orientación objetivo
                rotation_command = Twist()
                rotation_command.linear.x = 0.0
                rotation_command.angular.z = (self.final_rotation_speed if angular_error > 0 
                                            else -self.final_rotation_speed)
                self.cmd_vel_pub.publish(rotation_command)
                return

        # === Cálculo de comandos de velocidad con Pure Pursuit ===
        velocity_command = self.follow_path(robot_x, robot_y, estimated_pose, target_point)

        # === Limitación de curvatura para seguridad ===
        if abs(velocity_command.linear.x) > 1e-3:  # Evitar división por cero
            curvature = abs(velocity_command.angular.z / velocity_command.linear.x)
            if curvature > self.max_curvature:
                # Escalar velocidad para respetar curvatura máxima
                scale_factor = self.max_curvature / curvature
                velocity_command.linear.x *= scale_factor
            
            # Garantizar velocidad mínima para evitar atorarse
            velocity_command.linear.x = max(velocity_command.linear.x, 0.05)

        # === Enviar comando al robot ===
        self.cmd_vel_pub.publish(velocity_command)



    def handle_avoiding_obstacle_state(self, estimated_pose):
        """
        Maneja el estado de evasión de obstáculos mediante maniobra de giro.
        
        Estrategia:
        - Gira hacia la dirección que conduce al siguiente waypoint
        - Mantiene rotación hasta que el obstáculo no esté en el frente
        - Decide dirección de giro según ubicación del objetivo
        
        Transiciones:
        - AVOIDING_OBSTACLE → NAVIGATING: Cuando el frente queda libre
        
        Args:
            estimated_pose: Pose actual estimada del robot
        """
        
        # === Determinar dirección de giro en la primera ejecución ===
        if not hasattr(self, 'turn_direction'):
            robot_x = estimated_pose.position.x
            robot_y = estimated_pose.position.y
            robot_yaw = self.get_yaw_from_pose(estimated_pose)
            
            # Buscar próximo waypoint como referencia para dirección de giro
            target_point = self.find_lookahead_point(self.current_path, robot_x, robot_y)
            
            if target_point is None:
                # Si no hay waypoints, usar objetivo final como referencia
                if self.goal_pose is not None:
                    target_x = self.goal_pose.position.x
                    target_y = self.goal_pose.position.y
                else:
                    # Estrategia de respaldo: giro hacia la izquierda
                    self.turn_direction = 1
                    return
            else:
                target_x = target_point.position.x
                target_y = target_point.position.y
            
            # === Calcular dirección óptima de giro ===
            # Vector hacia el objetivo
            delta_x = target_x - robot_x
            delta_y = target_y - robot_y
            target_bearing = math.atan2(delta_y, delta_x)
            
            # Diferencia angular más corta hacia el objetivo
            angular_error = self.angle_diff(target_bearing, robot_yaw)
            
            # Seleccionar dirección de giro según ubicación del objetivo
            if angular_error > 0:
                self.turn_direction = 1  # Giro antihorario (izquierda)
            else:
                self.turn_direction = -1  # Giro horario (derecha)

        # === Ejecutar maniobra de giro ===
        evasion_command = Twist()
        evasion_command.linear.x = 0.0  # Detener movimiento hacia adelante
        evasion_command.angular.z = self.turn_direction * self.obstacle_avoidance_turn_speed
        self.cmd_vel_pub.publish(evasion_command)

        # === Verificar condición de salida ===
        if not self.is_obstacle_detected():
            self.get_logger().info("Obstáculo evadido - Retomando navegación")
            
            # Limpiar variables de estado de evasión
            if hasattr(self, 'turn_direction'):
                delattr(self, 'turn_direction')
            self.obstacle_avoidance_cumulative_angle = 0.0
            
            # Retornar a navegación normal
            self.state = State.NAVIGATING

    # --------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 