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

        # --- Parameters ---
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('initial_pose_topic', '/initialpose')
        self.declare_parameter('laser_max_range', 3.5)
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('obstacle_detection_distance', 0.6)
        self.declare_parameter('obstacle_avoidance_turn_speed', 0.5)
        self.declare_parameter('angular_gain', 1.5)  # Ganancia P para el ángulo
        self.declare_parameter('max_curvature', 3.0) 


        # --- Parameters to set ---
        # TODO: Setear valores default
        self.declare_parameter('num_particles', 100)                # Cantidad de partículas (balance entre precisión y velocidad)
        self.declare_parameter('alpha1', 0.005)                       # Ruido rotacional al rotar (rad²)
        self.declare_parameter('alpha2', 0.005)                       # Ruido rotacional al trasladar (rad²/m²)
        self.declare_parameter('alpha3', 0.01)                       # Ruido translacional al trasladar (m²/m²)
        self.declare_parameter('alpha4', 0.01)                       # Ruido translacional al rotar (m²/rad²)
        self.declare_parameter('z_hit', 0.9)                        # Peso de coincidencia entre LIDAR y mapa
        self.declare_parameter('z_rand', 0.1)                       # Peso de lecturas aleatorias (ruido del LIDAR)
        self.declare_parameter('lookahead_distance', 0.4)           # Para el seguimiento del camino (distancia de anticipación)
        self.declare_parameter('linear_velocity', 0.2)              # Velocidad base de navegación (m/s)
        self.declare_parameter('goal_tolerance', 0.15)              # Tolerancia para considerar que llegó al objetivo (m)
        self.declare_parameter('path_pruning_distance', 0.3)        # Distancia para podar puntos viejos del camino (m)
        self.declare_parameter('safety_margin_cells', 4)            # Celdas a expandir alrededor de obstáculos para seguridad


                
        
        self.num_particles = self.get_parameter('num_particles').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.laser_max_range = self.get_parameter('laser_max_range').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        self.alphas = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_pruning_distance = self.get_parameter('path_pruning_distance').value
        self.safety_margin_cells = self.get_parameter('safety_margin_cells').value
        self.obstacle_detection_distance = self.get_parameter('obstacle_detection_distance').value
        self.obstacle_avoidance_turn_speed = self.get_parameter('obstacle_avoidance_turn_speed').value
        self.max_curvature = self.get_parameter('max_curvature').value
        self.angular_gain = self.get_parameter('angular_gain').value

        # --- State ---
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.map_data = None
        self.latest_scan = None
        self.initial_pose_received = False
        self.map_received = False
        self.last_odom_pose = None
        self.state = State.IDLE
        self.current_path = None
        self.goal_pose = None
        self.inflated_grid = None
        self.obstacle_avoidance_start_yaw = None
        self.obstacle_avoidance_last_yaw = None
        self.obstacle_avoidance_cumulative_angle = 0.0
        self.obstacle_avoidance_active = False
        
        # --- ROS 2 Interfaces ---
        map_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        scan_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        
        self.map_sub = self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, scan_qos)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.get_parameter('initial_pose_topic').value, self.initial_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)
        
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 10)
        self.particle_pub = self.create_publisher(MarkerArray, 'particle_cloud', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('MyPyAMCL node initialized.')

    def map_callback(self, msg):
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            self.inflate_map()
            self.get_logger().info('Map and inflated map processed.')

            # 1) distancia euclídea a la celda ocupada más cercana
            occupied = self.grid > 50
            # La edt da distancia en celdas ⇒ multiplica por resolución para obtener metros
            self.dist_field = distance_transform_edt(~occupied).astype(np.float32) * self.map_data.info.resolution
            self.get_logger().info("Likelihood field pre-calculado.")


    def scan_callback(self, msg):
        self.latest_scan = msg

    def goal_callback(self, msg):
        if self.map_data is None:
            self.get_logger().warn("Goal received, but map is not available yet. Ignoring goal.")
            return

        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Goal received in frame '{msg.header.frame_id}', but expected '{self.map_frame_id}'. Ignoring.")
            return
            
        self.goal_pose = msg.pose
        self.get_logger().info(f"New goal received: ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f}). State -> PLANNING")
        self.state = State.PLANNING
        self.current_path = None

    def initial_pose_callback(self, msg):
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Initial pose frame is '{msg.header.frame_id}' but expected '{self.map_frame_id}'. Ignoring.")
            return
        self.get_logger().info('Initial pose received.')
        self.initialize_particles(msg.pose.pose)
        self.publish_pose(msg.pose.pose)
        
        odom_tf = self.get_odom_transform()
        if odom_tf is not None:
            self.publish_transform(msg.pose.pose, odom_tf)
        else:
            self.get_logger().warn("No se pudo obtener odom→base para publicar el transform inicial.")
        
        self.state = State.IDLE
        self.initial_pose_received = True
        self.last_odom_pose = None # Reset odom tracking
        self.stop_robot()



    def initialize_particles(self, initial_pose):
        # TODO: Inicializar particulas en base a la pose inicial con variaciones aleatorias
        # Deben ser la misma cantidad de particulas que self.num_particles
        # Deben tener un peso

        # Centro de las partículas: la pose inicial recibida
        mean_x = initial_pose.position.x
        mean_y = initial_pose.position.y

        # Convertir la orientación en cuaternión a yaw (ángulo en el plano)
        q = initial_pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]

        # Parámetros de ruido: desviaciones estándar
        std = [0.2, 0.2, 0.1]  # 20 cm de posición, 0.1 rad de orientación

        # Generar partículas alrededor de la pose inicial
        self.particles[:, 0] = np.random.normal(mean_x, std[0], self.num_particles)  # x
        self.particles[:, 1] = np.random.normal(mean_y, std[1], self.num_particles)  # y
        self.particles[:, 2] = np.random.normal(yaw, std[2], self.num_particles)     # yaw

        # Inicializar pesos uniformes
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Publicar en RViz para visualización
        self.publish_particles()


    def initialize_particles_randomly(self):
        # TODO: Inizializar particulas aleatoriamente en todo el mapa
        self.get_logger().info("Inicializando partículas aleatoriamente (sin pose inicial).")
        free_indices = np.where(np.array(self.map_data.data) == 0)[0]

        if len(free_indices) == 0:
            self.get_logger().warn("No hay celdas libres para inicializar partículas.")
            return

        chosen_indices = np.random.choice(free_indices, self.num_particles)
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height

        xs = []
        ys = []

        for idx in chosen_indices:
            gy = idx // map_w
            gx = idx % map_w
            wx, wy = self.grid_to_world(gx, gy)
            xs.append(wx)
            ys.append(wy)

        self.particles[:, 0] = xs
        self.particles[:, 1] = ys
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()


    def timer_callback(self):
        # --- Debug logs ---
        self.get_logger().info(f"[TIMER] map_received: {self.map_received}")
        self.get_logger().info(f"[TIMER] latest_scan: {self.latest_scan is not None}")
        self.get_logger().info(f"[TIMER] initial_pose_received: {self.initial_pose_received}")

        # --- Preconditions ---
        if not self.map_received:
            return
        if self.latest_scan is None:
            return
        if not self.initial_pose_received:
            return

        # --- Get odom->base transform ---
        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            self.stop_robot()
            return

        # --- Localization update ---
        estimated_pose = self.run_localization_cycle(current_odom_tf)

        # --- Publish state estimates ---
        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)

        # --- State machine ---
        if self.state == State.IDLE:
            # Remain stopped until a new goal is received
            self.stop_robot()
            return

        if self.state == State.PLANNING:
            # Run A* planning; transitions to NAVIGATING internally
            self.handle_planning_state(estimated_pose)
            return

        if self.state == State.NAVIGATING:
            # Follow path with Pure Pursuit; may transition to IDLE or AVOIDING_OBSTACLE
            self.handle_navigating_state(estimated_pose)
            return

        if self.state == State.AVOIDING_OBSTACLE:
            # Execute obstacle avoidance maneuver; transitions back to NAVIGATING
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
        Modelo de movimiento «sample_motion_model_odometry» pulido:
        - Distingue giro-in-place de desplazamiento
        - Varianzas cuadráticas según Thrun (Prob. Robotics, Eq. 5.6)
        - Normalización de ángulos
        """
        self.get_logger().info("Actualizando partículas con modelo de movimiento pulido.")
        # 1) Si es la primera llamada, solo guardamos odom
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_tf.transform
            return

        # 2) Delta traslación
        t1 = self.last_odom_pose.translation
        t2 = current_odom_tf.transform.translation
        dx, dy = t2.x - t1.x, t2.y - t1.y
        delta_trans = np.hypot(dx, dy)

        # 3) Delta rotaciones
        q1 = self.last_odom_pose.rotation
        q2 = current_odom_tf.transform.rotation
        theta1 = R.from_quat([q1.x, q1.y, q1.z, q1.w]).as_euler('zyx')[0]
        theta2 = R.from_quat([q2.x, q2.y, q2.z, q2.w]).as_euler('zyx')[0]

        def normalize(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        if delta_trans < 1e-4:
            # Giro en el mismo punto
            delta_rot1 = 0.0
            delta_rot2 = normalize(theta2 - theta1)
        else:
            delta_rot1 = normalize(np.arctan2(dy, dx) - theta1)
            delta_rot2 = normalize((theta2 - theta1) - delta_rot1)

        # 4) Si no hubo movimiento real, saltamos todo el ruido
        if delta_trans < 1e-4 and abs(normalize(theta2 - theta1)) < 1e-4:
            # Solo avanzamos el timestamp
            self.last_odom_pose = current_odom_tf.transform
            return

        # 5) Cálculo de varianzas (Thrun, Eq. 5.6)
        a1, a2, a3, a4 = self.alphas
        var_r1 = a1 * delta_rot1**2 + a2 * delta_trans**2
        var_t  = a3 * delta_trans**2 + a4 * (delta_rot1**2 + delta_rot2**2)
        var_r2 = a1 * delta_rot2**2 + a2 * delta_trans**2

        # 6) Muestras de movimiento ruidoso (vectorizado)
        delta_rot1_hat = delta_rot1 + np.random.normal(0, np.sqrt(var_r1), self.num_particles)
        delta_trans_hat = delta_trans + np.random.normal(0, np.sqrt(var_t),  self.num_particles)
        delta_rot2_hat = delta_rot2 + np.random.normal(0, np.sqrt(var_r2), self.num_particles)

        # 7) Actualización de partículas
        self.particles[:, 0] += delta_trans_hat * np.cos(self.particles[:, 2] + delta_rot1_hat)
        self.particles[:, 1] += delta_trans_hat * np.sin(self.particles[:, 2] + delta_rot1_hat)
        self.particles[:, 2]  = normalize(self.particles[:, 2] + delta_rot1_hat + delta_rot2_hat)

        # 8) Guardar odometría para la próxima iteración
        self.last_odom_pose = current_odom_tf.transform


    def measurement_model(self):
        if self.latest_scan is None or self.map_data is None:
            return

        desv_hit   = 0.12      # desviación típica del láser (~12 cm)
        λ_short = 0.1       # lambda para lecturas cortas (opcional)
        z_hit   = 0.80      # mezcla de modelos (deja 0.05–0.15 para z_rand)
        z_rand  = 0.20
        max_r   = self.laser_max_range

        ranges  = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        angles  = np.arange(self.latest_scan.angle_min,
                            self.latest_scan.angle_max,
                            self.latest_scan.angle_increment,
                            dtype=np.float32)

        # Sub-muestreo para mantener tiempo real
        step = 8                          # usa 360/8 ≃ 45 rayos por scan
        ranges = ranges[::step]
        angles = angles[::step]

        weights = np.zeros(self.num_particles, dtype=np.float32)

        for i, (px, py, pθ) in enumerate(self.particles):
            # Pre-genera senos y cosenos para todos los ángulos de este partícula-scan
            sin_a = np.sin(pθ + angles)
            cos_a = np.cos(pθ + angles)

            # Coordenadas (x,y) de cada endpoint según la partícula
            ex = px + ranges * cos_a
            ey = py + ranges * sin_a

            # A índice de celda
            gx = ((ex - self.map_data.info.origin.position.x) /
                self.map_data.info.resolution).astype(np.int32)
            gy = ((ey - self.map_data.info.origin.position.y) /
                self.map_data.info.resolution).astype(np.int32)

            inside = (gx >= 0) & (gx < self.map_data.info.width) & \
                    (gy >= 0) & (gy < self.map_data.info.height) & \
                    ~np.isnan(ranges)

            # Distancia al obstáculo más cercano (metros)
            d = np.full_like(ranges, max_r, dtype=np.float32)
            valid_idx = inside.nonzero()[0]
            d[valid_idx] = self.dist_field[gy[valid_idx], gx[valid_idx]]

            # Modelo de probabilidad p(z | x)
            p_hit   = z_hit  * np.exp(-0.5 * (d / desv_hit)**2) / (desv_hit * np.sqrt(2*np.pi))
            p_rand  = z_rand / max_r
            prob    = p_hit + p_rand            # podemos añadir p_short o p_max si quieres

            # Convertir a log-prob para evitar underflow y sumar
            weights[i] = np.exp(np.sum(np.log(prob + 1e-12)))

        # Normaliza y evita división por cero
        weights += 1e-300
        self.weights = weights / np.sum(weights)


    def resample(self):
        # TODO: Implementar el resampleo de las particulas basado en los pesos.
        self.get_logger().info("Resampleo de partículas.")
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles



    def estimate_pose(self):
        # TODO: Implementar la estimación de pose a partir de las particulas y sus pesos.
        
        # Media ponderada de las partículas
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)

        # Promedio circular para el ángulo
        sin_sum = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_sum = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        yaw = np.arctan2(sin_sum, cos_sum)

        # Crear objeto Pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        q = R.from_euler('z', yaw).as_quat()
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return pose

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
        map_to_base_mat = self.pose_to_matrix(estimated_pose)                         # Transformación estimada entre map → base
        odom_to_base_mat = self.transform_to_matrix(odom_tf.transform)                # Transformación real odometría odom → base
        map_to_odom_mat = np.dot(map_to_base_mat, np.linalg.inv(odom_to_base_mat))    # Queremos: map → odom = map → base × inv(odom → base)
        
        t = TransformStamped()
        
        # TODO: Completar el TransformStamped con la transformacion entre el mapa y la base del robot.
        # Extraer traslación y rotación
        translation = map_to_odom_mat[:3, 3]
        rotation = R.from_matrix(map_to_odom_mat[:3, :3]).as_quat()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id  # "map"
        t.child_frame_id = self.odom_frame_id  # "odom"
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation = Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])

        self.get_logger().info(f"Publicando transform map → odom: [{translation[0]:.2f}, {translation[1]:.2f}]")
        self.tf_broadcaster.sendTransform(t)

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

    # ------- FUNCIONES AUXILIARES ---------------

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def get_yaw_from_pose(self, pose):
        q = pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
        return yaw

    def angle_diff(self, a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

    def is_obstacle_detected(self):
        if self.latest_scan is None:
            return False

        for r in self.latest_scan.ranges:
            if not np.isnan(r) and r < self.obstacle_detection_distance:
                return True
        return False
    
    def astar(self, start, goal):
            """
            start, goal: tuplas (gx, gy) en celdas.
            Usa self.inflated_grid para obstáculos.
            """
            # Helpers
            def h(a, b):
                return abs(a[0]-b[0]) + abs(a[1]-b[1])

            # Nodos abiertos: (f = g+h, g, (x,y), parent)
            open_heap = [(h(start, goal), 0, start, None)]
            came_from  = {}
            cost_so_far = {start: 0}

            while open_heap:
                f, g, current, parent = heapq.heappop(open_heap)
                if current in came_from:
                    continue
                came_from[current] = parent
                if current == goal:
                    # reconstruir
                    path = []
                    node = current
                    while node:
                        path.append(node)
                        node = came_from[node]
                    return list(reversed(path))

                # vecinos 4-conectados
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = current[0]+dx, current[1]+dy
                    if (0 <= nx < self.map_data.info.width and
                        0 <= ny < self.map_data.info.height and
                        self.inflated_grid[ny, nx] == 0):
                        new_cost = g + 1
                        if (nx,ny) not in cost_so_far or new_cost < cost_so_far[(nx,ny)]:
                            cost_so_far[(nx,ny)] = new_cost
                            priority = new_cost + h((nx,ny), goal)
                            heapq.heappush(open_heap, (priority, new_cost, (nx,ny), current))
            return None

    def plan_path(self, start, goal):
        # Antes de nada, imprime el origen y la resolución del mapa
        o = self.map_data.info.origin.position
        res = self.map_data.info.resolution
        self.get_logger().info(f"🗺️ Map origin: ({o.x:.2f},{o.y:.2f}), res: {res:.2f}")

        # Convertir a grid
        start_gx, start_gy = self.world_to_grid(start.x, start.y)
        goal_gx, goal_gy = self.world_to_grid(goal.x, goal.y)
        self.get_logger().info(f"🗺️ plan_path // start_grid: ({start_gx},{start_gy})  goal_grid: ({goal_gx},{goal_gy})")

        # Validación
        if not (0 <= start_gx < self.grid.shape[1] and 0 <= start_gy < self.grid.shape[0]):
            self.get_logger().warn(f"Start fuera de rango: ({start_gx},{start_gy})")
            return None
        if not (0 <= goal_gx < self.grid.shape[1] and 0 <= goal_gy < self.grid.shape[0]):
            self.get_logger().warn(f"Goal fuera de rango: ({goal_gx},{goal_gy})")
            return None

        # A* path planning
        path_cells = self.astar((start_gx, start_gy), (goal_gx, goal_gy))
        if not path_cells:
            self.get_logger().warn("A* no encontró ruta.")
            return None

        # Convertir a Path msg
        path_msg = Path()
        for gx, gy in path_cells:
            wx, wy = self.grid_to_world(gx, gy)
            pose = PoseStamped()
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

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
        obstacle_mask = self.grid > 50
        kernel = np.ones((2 * self.safety_margin_cells + 1, 2 * self.safety_margin_cells + 1))
        inflated = grey_dilation(obstacle_mask.astype(np.uint8), footprint=kernel)
        self.inflated_grid = inflated * 100
        self.get_logger().info("Mapa inflado con dilatación real.")

    def run_localization_cycle(self, current_odom_tf):
        self.motion_model(current_odom_tf)
        self.measurement_model()
        self.resample()
        return self.estimate_pose()
    

    def handle_planning_state(self, estimated_pose):
        """
        Ejecuta la planificación de ruta con A* y publica el camino.
        Transición:
          PLANNING -> NAVIGATING (si hay ruta)
          PLANNING -> IDLE       (si falla la planificación)
        """
        # Verificar que exista un goal válido
        if self.goal_pose is None:
            self.get_logger().warn("PLANNING state but goal_pose is None. Transitioning to IDLE.")
            self.state = State.IDLE
            return

        # Llamar al planificador A*
        path_msg = self.plan_path(estimated_pose.position, self.goal_pose.position)
        # Si no se encontró ruta o está vacía
        if path_msg is None or len(path_msg.poses) == 0:
            self.get_logger().warn("Planning failed, no path found. Transitioning to IDLE.")
            self.state = State.IDLE
            self.stop_robot()
            return

        # Publicar el path en RViz y guardar el camino
        # publish_path() añade header y timestamp automáticamente
        self.current_path = path_msg
        self.publish_path(path_msg)

        self.get_logger().info(f"Path planned with {len(path_msg.poses)} waypoints. Transitioning to NAVIGATING.")
        self.state = State.NAVIGATING



    def handle_navigating_state(self, estimated_pose):
        """
        Control en estado NAVIGATING: poda de path, detección de llegada,
        cálculo de pure pursuit, y escalado de velocidad según curvatura.
        """
        # 1) Obtener posición actual del robot
        robot_x = estimated_pose.position.x
        robot_y = estimated_pose.position.y

        # 2) Detección de llegada usando goal_tolerance
        dxg = self.goal_pose.position.x - robot_x
        dyg = self.goal_pose.position.y - robot_y
        if np.hypot(dxg, dyg) < self.goal_tolerance:
            self.get_logger().info("Objetivo alcanzado. State -> IDLE")
            self.state = State.IDLE
            self.stop_robot()
            return

        # 3) Poda de waypoints ya recorridos (path_pruning_distance)
        self.current_path.poses = [
            p for p in self.current_path.poses
            if np.hypot(
                p.pose.position.x - robot_x,
                p.pose.position.y - robot_y
            ) > self.path_pruning_distance
        ]

        # 4) Búsqueda del punto look-ahead
        target = self.find_lookahead_point(self.current_path, robot_x, robot_y)
        if target is None:
            self.get_logger().info("Objetivo alcanzado o sin más waypoints. State -> IDLE")
            self.state = State.IDLE
            self.stop_robot()
            return

        # 5) Cálculo del Twist base con Pure Pursuit
        cmd = self.follow_path(robot_x, robot_y, estimated_pose, target)

        curvature = cmd.angular.z / max(1e-3, cmd.linear.x)
        v_scale = min(1.0, self.max_curvature / abs(curvature))
        cmd.linear.x *= v_scale
        cmd.linear.x = max(cmd.linear.x, 0.05)  # al menos 5 cm/s

        # 6) Publicar comando
        self.cmd_vel_pub.publish(cmd)



    def handle_avoiding_obstacle_state(self, estimated_pose):
        """
        AVOIDING_OBSTACLE: giro en sitio hasta acumular 90° y luego NAVIGATING.
        """
        # 1) Calcular yaw actual y cuánto giró desde la última iteración
        yaw_now = self.get_yaw_from_pose(estimated_pose)
        delta = self.angle_diff(yaw_now, self.obstacle_avoidance_last_yaw)
        self.obstacle_avoidance_cumulative_angle += abs(delta)
        self.obstacle_avoidance_last_yaw = yaw_now

        # 2) Publicar sólo giro (sin avance lineal)
        twist = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = self.obstacle_avoidance_turn_speed
        self.cmd_vel_pub.publish(twist)

        # 3) Si ya giró ≥ 90°, terminar evasión
        if self.obstacle_avoidance_cumulative_angle >= (math.pi / 2):
            self.get_logger().info(
                f"Finished avoidance ({self.obstacle_avoidance_cumulative_angle:.2f} rad). → NAVIGATING"
            )
            # Reset y transición
            self.obstacle_avoidance_cumulative_angle = 0.0
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