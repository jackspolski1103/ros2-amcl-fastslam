#!/usr/bin/env python3
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
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convierte ángulos de Euler a quaternion de forma robusta
    Args:
        roll, pitch, yaw: ángulos en radianes
    Returns:
        tuple: (x, y, z, w) componentes del quaternion normalizado
    """
    # Normalizar ángulos de entrada para evitar acumulación de errores
    roll = normalize_angle(roll)
    pitch = normalize_angle(pitch)
    yaw = normalize_angle(yaw)
    
    # Calcular componentes con mayor precisión
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    # Normalizar quaternion resultante para evitar drift numérico
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm > 1e-10:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    else:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    
    return x, y, z, w

class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)
        # Almacenar la odometría anterior de esta partícula para FastSLAM
        self.prev_odom_x = 0.0
        self.prev_odom_y = 0.0
        self.prev_odom_theta = 0.0

    def pose(self):
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node):
    def __init__(self):
        super().__init__('python_slam_node')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        # TODO: define map resolution, width, height, and number of particles
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_meters', 6.0)
        self.declare_parameter('map_height_meters', 6.0)
        self.declare_parameter('num_particles', 10)

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -2.5
        self.map_origin_y = -5.0

        # TODO: define the log-odds criteria for free and occupied cells
        # Smaller increments for more stable mapping
        self.log_odds_free = -0.05
        self.log_odds_occupied = 0.10
        self.log_odds_max = 10.0
        self.log_odds_min = -10.0

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.last_odom = None
        self.current_map_pose = [0.0, 0.0, 0.0]
        self.current_odom_pose = [0.0, 0.0, 0.0]

        # Motion noise parameters - VALORES REDUCIDOS
        self.alpha1 = 0.005  # rotation -> rotation noise (era 0.1)
        self.alpha2 = 0.005  # translation -> rotation noise (era 0.1)
        self.alpha3 = 0.005  # translation -> translation noise (era 0.1)
        self.alpha4 = 0.005  # rotation -> translation noise (era 0.1)
        
        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        # self.particle_pub = self.create_publisher(MarkerArray, 'particle_cloud', 10)
        # self.best_particle_scan_pub = self.create_publisher(LaserScan, '/best_particle_scan', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # === Manejo de transformaciones ===
        self.tf_buffer = Buffer()  # Buffer para almacenar transformaciones recientes
        self.tf_listener = TransformListener(self.tf_buffer, self)  # Escucha transformaciones del árbol TF
        
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

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            return

        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        odom_x = odom.pose.pose.position.x
        odom_y = odom.pose.pose.position.y
        quaternion = odom.pose.pose.orientation
        _, _, odom_theta = euler_from_quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.current_odom_pose = [odom_x, odom_y, odom_theta]

        # Motion update only if we have previous odometry
        if odom is not None:
            # FastSLAM: Para cada partícula, calcular movimiento desde SU odometría anterior
            for p in self.particles:
                # Calcular diferencia desde la odometría anterior de ESTA partícula
                delta_x = odom_x - p.prev_odom_x
                delta_y = odom_y - p.prev_odom_y
                delta_theta = self.angle_diff(odom_theta, p.prev_odom_theta)
                
                # Apply motion model with noise
                noise_x = np.random.normal(0, self.alpha3 * abs(delta_x) + self.alpha4 * abs(delta_theta))
                noise_y = np.random.normal(0, self.alpha3 * abs(delta_y) + self.alpha4 * abs(delta_theta))
                noise_theta = np.random.normal(0, self.alpha1 * abs(delta_theta) + self.alpha2 * (abs(delta_x) + abs(delta_y)))
                
                p.x += delta_x + noise_x
                p.y += delta_y + noise_y
                p.theta = normalize_angle(p.theta + delta_theta + noise_theta)
                
                # Actualizar la odometría anterior de esta partícula
                p.prev_odom_x = odom_x
                p.prev_odom_y = odom_y
                p.prev_odom_theta = odom_theta

        # TODO: 2. Measurement update (weight particles)
        weights = []
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(self.particles) for _ in range(len(self.particles))]

        for i, p in enumerate(self.particles):
            p.weight = weights[i] # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        weighted_x = sum(p.x * p.weight for p in self.particles)
        weighted_y = sum(p.y * p.weight for p in self.particles)
        weighted_theta = sum(p.theta * p.weight for p in self.particles)
        self.current_map_pose = [weighted_x, weighted_y, weighted_theta]

        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom()

        #  (extra) Publish visualizations
        # self.publish_particles()
        # self.publish_best_particle_scan(msg)

    def compute_weight(self, particle, scan_msg):
        """
        Calcula el peso de una partícula de la manera más simple posible:
        Solo cuenta cuántos endpoints del laser coinciden con celdas ocupadas
        """
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        
        hit_count = 0
        total_valid_readings = 0
        
        for i, range_dist in enumerate(scan_msg.ranges):
            # Filtrar lecturas válidas
            if (math.isnan(range_dist) or 
                range_dist < scan_msg.range_min or 
                range_dist >= scan_msg.range_max):
                continue
            
            # Calcular endpoint del rayo láser
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            endpoint_x = robot_x + range_dist * math.cos(robot_theta + angle)
            endpoint_y = robot_y + range_dist * math.sin(robot_theta + angle)
            
            # Convertir a coordenadas del mapa
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
        # SUS (Stochastic Universal Sampling) - método más simple
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
        
        # Resampling
        new_particles = []
        for pointer in pointers:
            # Encontrar índice donde cumulative >= pointer
            idx = np.searchsorted(cumulative, pointer)
            if idx >= len(particles):
                idx = len(particles) - 1
            
            # Copiar partícula seleccionada
            p = particles[idx]
            new_particle = Particle(p.x, p.y, p.theta, 1.0/len(particles), 
                                  (self.map_height_cells, self.map_width_cells))
            new_particle.log_odds_map = p.log_odds_map.copy()
            new_particle.prev_odom_x = p.prev_odom_x
            new_particle.prev_odom_y = p.prev_odom_y
            new_particle.prev_odom_theta = p.prev_odom_theta
            new_particles.append(new_particle)
        
        return new_particles

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue
            # TODO: Update map: transform the scan into the map frame
            angle =  i * scan_msg.angle_increment - scan_msg.angle_min 
            endpoint_x = robot_x + current_range * math.cos(robot_theta + angle)
            endpoint_y = robot_y + current_range * math.sin(robot_theta + angle)
            
            # Convert to map coordinates
            robot_map_x = int((robot_x - self.map_origin_x) / self.resolution)
            robot_map_y = int((robot_y - self.map_origin_y) / self.resolution)
            endpoint_map_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            endpoint_map_y = int((endpoint_y - self.map_origin_y) / self.resolution)

            # TODO: Use self.bresenham_line for free cells
            self.bresenham_line(particle, robot_map_x, robot_map_y, endpoint_map_x, endpoint_map_y)

            # TODO: Update particle.log_odds_map accordingly
            # Mark endpoint as occupied if it's a valid hit
            if is_hit and 0 <= endpoint_map_x < self.map_width_cells and 0 <= endpoint_map_y < self.map_height_cells:
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] += self.log_odds_occupied
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] = np.clip(particle.log_odds_map[endpoint_map_y, endpoint_map_x], self.log_odds_min, self.log_odds_max)

    def bresenham_line(self, particle, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(particle.log_odds_map[y0, x0], self.log_odds_min, self.log_odds_max)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        # TODO: Fill in map_msg fields and publish one map
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value

        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Use the best particle (highest weight) for the published map
        best_particle = max(self.particles, key=lambda p: p.weight)

        # Convert log-odds to occupancy grid using NumPy
        log_odds_map = best_particle.log_odds_map
        occupancy_grid = np.full_like(log_odds_map, -1, dtype=np.int8)
        occupancy_grid[log_odds_map > 0.5] = 100  # Occupied
        occupancy_grid[log_odds_map < -0.5] = 0    # Free

        map_msg.data = occupancy_grid.flatten().tolist()

        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")

    def get_odom_transform(self):
        """
        Obtiene la transformación actual de odometría (odom → base_link).
        
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
        Convierte una pose a matriz de transformación homogénea 4x4.
        
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
        Convierte una transformación a matriz homogénea 4x4.
        
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
        Publica la transformación map → odom necesaria para el árbol de transformaciones.
        
        Cálculo:
        map → odom = map → base_link × inv(odom → base_link)
        
        Esta transformación permite que el árbol TF sea consistente:
        map → odom → base_link
        """
        # === Obtener transformación actual de odometría ===
        odom_tf = self.get_odom_transform()
        if odom_tf is None:
            self.get_logger().warn("No se puede publicar transformación map→odom: falta transformación odom→base_link")
            return
        
        # === Crear pose estimada del robot en el frame del mapa ===
        estimated_pose = Pose()
        estimated_pose.position.x = self.current_map_pose[0]
        estimated_pose.position.y = self.current_map_pose[1]
        estimated_pose.position.z = 0.0
        
        # Convertir ángulo yaw a cuaternión
        quaternion = R.from_euler('z', self.current_map_pose[2]).as_quat()
        estimated_pose.orientation = Quaternion(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )
        
        # === Calcular transformación map → odom usando matrices homogéneas ===
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
        transform_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        transform_msg.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
        transform_msg.transform.translation.x = translation[0]
        transform_msg.transform.translation.y = translation[1]
        transform_msg.transform.translation.z = translation[2]
        transform_msg.transform.rotation = Quaternion(
            x=rotation_quat[0], y=rotation_quat[1], z=rotation_quat[2], w=rotation_quat[3]
        )

        # === Publicar transformación al árbol TF ===
        self.tf_broadcaster.sendTransform(transform_msg)



    # def publish_particles(self):
    #     """
    #     Publishes particle cloud as MarkerArray for visualization in RViz
    #     """
    #     ma = MarkerArray()
    #     for i, particle in enumerate(self.particles):
    #         marker = Marker()
    #         marker.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
    #         marker.header.stamp = self.get_clock().now().to_msg()
    #         marker.ns = "particles"
    #         marker.id = i
    #         marker.type = Marker.ARROW
    #         marker.action = Marker.ADD
    #         marker.pose.position.x = particle.x
    #         marker.pose.position.y = particle.y
    #         marker.pose.position.z = 0.0
            
    #         # Convert theta to quaternion
    #         quat_x, quat_y, quat_z, quat_w = quaternion_from_euler(0, 0, particle.theta)
    #         marker.pose.orientation.x = quat_x
    #         marker.pose.orientation.y = quat_y
    #         marker.pose.orientation.z = quat_z
    #         marker.pose.orientation.w = quat_w
            
    #         # Scale based on particle weight
    #         base_scale = 0.1
    #         weight_scale = max(0.5, particle.weight * self.num_particles * 2.0)  # Scale by weight
    #         marker.scale.x = base_scale * weight_scale
    #         marker.scale.y = 0.02 * weight_scale
    #         marker.scale.z = 0.02 * weight_scale
            
    #         # Color based on weight (red = high weight, blue = low weight)
    #         marker.color.a = 0.7
    #         marker.color.r = min(1.0, particle.weight * self.num_particles * 2.0)
    #         marker.color.g = 0.2
    #         marker.color.b = max(0.3, 1.0 - particle.weight * self.num_particles * 2.0)
            
    #         ma.markers.append(marker)
        
    #     self.particle_pub.publish(ma)

    # def publish_best_particle_scan(self, original_scan):
    #     """
    #     Publishes the laser scan as seen from the best particle's pose
    #     This allows visualization of how the scan looks from the estimated robot position
    #     """
    #     # Find the best particle (highest weight)
    #     best_particle = max(self.particles, key=lambda p: p.weight)
        
    #     # Create a new LaserScan message
    #     best_scan = LaserScan()
    #     best_scan.header = original_scan.header
    #     best_scan.header.frame_id = "best_particle_scan"  # Custom frame for visualization
    #     best_scan.angle_min = original_scan.angle_min
    #     best_scan.angle_max = original_scan.angle_max
    #     best_scan.angle_increment = original_scan.angle_increment
    #     best_scan.time_increment = original_scan.time_increment
    #     best_scan.scan_time = original_scan.scan_time
    #     best_scan.range_min = original_scan.range_min
    #     best_scan.range_max = original_scan.range_max
        
    #     # Transform scan ranges based on best particle's position and map knowledge
    #     best_scan.ranges = []
        
    #     for i, original_range in enumerate(original_scan.ranges):
    #         # Use original range if it's valid
    #         if (not math.isnan(original_range) and 
    #             original_scan.range_min <= original_range < original_scan.range_max):
    #             best_scan.ranges.append(original_range)
    #         else:
    #             # For invalid ranges, try to estimate based on the map
    #             angle = original_scan.angle_min + i * original_scan.angle_increment
    #             estimated_range = self.estimate_range_from_map(best_particle, angle, original_scan.range_max)
    #             best_scan.ranges.append(estimated_range)
        
    #     # Publish the enhanced scan
    #     # self.best_particle_scan_pub.publish(best_scan)
        

    # def estimate_range_from_map(self, particle, angle, max_range):
    #     """
    #     Estimate the range for a laser beam based on the particle's map
    #     """
    #     robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        
    #     # Step along the ray until we hit an obstacle or reach max range
    #     step_size = self.resolution * 0.5  # Half cell size for better precision
    #     current_range = 0.0
        
    #     while current_range < max_range:
    #         # Calculate current point along the ray
    #         point_x = robot_x + current_range * math.cos(robot_theta + angle)
    #         point_y = robot_y + current_range * math.sin(robot_theta + angle)
            
    #         # Convert to map coordinates
    #         map_x = int((point_x - self.map_origin_x) / self.resolution)
    #         map_y = int((point_y - self.map_origin_y) / self.resolution)
            
    #         # Check if we're outside the map bounds
    #         if (map_x < 0 or map_x >= self.map_width_cells or 
    #             map_y < 0 or map_y >= self.map_height_cells):
    #             return max_range
            
    #         # Check if we hit an obstacle (occupied cell)
    #         if particle.log_odds_map[map_y, map_x] > 0:
    #             return current_range
            
    #         current_range += step_size
        
    #     return max_range


    @staticmethod
    def angle_diff(a, b):
        """
        Calcula la diferencia angular más corta entre dos ángulos de forma robusta
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