from __future__ import annotations

from shapely import GeometryCollection
from shapely.geometry import Polygon, mapping, LineString
from scipy.optimize import minimize
import numpy as np
from warnings import warn
from itertools import combinations
from random import sample

from quadrilateral_fitter import _Line 

class QuadrilateralFitter:
    def __init__(self, polygon: np.ndarray | tuple | list | Polygon):
        """
        Constructor for initializing the QuadrilateralFitter object.

        :param polygon: np.ndarray. A NumPy array of shape (N, 2) representing the input polygon,
                              where N is the number of vertices.
        """
        
        if isinstance(polygon, Polygon):
            _polygon = polygon
            self._polygon_coords = np.array(polygon.exterior.coords, dtype=np.float32)
        else:
            if type(polygon) == np.ndarray:
                assert polygon.shape[1] == len(
                    polygon.shape) == 2, f"Input polygon must have shape (N, 2). Got {polygon.shape}"
                _polygon = Polygon(polygon)
                self._polygon_coords = polygon
            elif isinstance(polygon, (list, tuple)):
                # Checking if the list or tuple has sub-lists/tuples of length 2 (i.e., coordinates)
                assert all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in
                           polygon), "Expected sub-lists or sub-tuples of length 2 for coordinates"
                _polygon = Polygon(polygon)
                self._polygon_coords = np.array(polygon, dtype=np.float32)
            else:
                raise TypeError(f"Unexpected input type: {type(polygon)}. Accepted are np.ndarray, tuple, "
                                f"list and shapely.Polygon")

            if isinstance(_polygon, LineString):
                warn("Polygon coordinates casted to a LineString. Quadrilateral Fitting results may be inaccurate.")
                # Extract the line coordinates
                line_coords = np.array(_polygon.coords, dtype=np.float32)

                # Well define a rectangle around it with a margin
                min_x, min_y = np.min(line_coords, axis=0)
                max_x, max_y = np.max(line_coords, axis=0)
                margin = 10

                # Define the new coordinates for the tight rectangle with margin
                false_coords = [
                    [min_x - margin, min_y - margin],
                    [min_x - margin, max_y + margin],
                    [max_x + margin, max_y + margin],
                    [max_x + margin, min_y - margin],
                    [min_x - margin, min_y - margin]
                ]

                _polygon = Polygon(false_coords)
                assert isinstance(_polygon, Polygon), "Expected a Polygon object from the input coordinates"

                self._polygon_coords = np.array(_polygon.exterior.coords, dtype=np.float32)

        self.convex_hull_polygon = _polygon.convex_hull
        self.centroid = self.convex_hull_polygon.centroid

        self._initial_guess = None

        self._line_equations = None
        self.fitted_quadrilateral = None

        self._expanded_line_equations = None
        self.expanded_fitted_quadrilateral = None

    def fit(self, simplify_polygons_larger_than: int|None = 10, start_simplification_epsilon: float = 0.1,
            max_simplification_epsilon: float = 0.5, simplification_epsilon_increment: float = 0.02,
            expansion_margin: float = 0.0) -> \
            tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Fits an irregular quadrilateral around the input polygon. The quadrilateral is optimized to minimize
        the Intersection over Union (IoU) with the input polygon.

        This method performs the following steps:
        1. Computes the convex hull of the input polygon.
        2. Finds an initial quadrilateral that closely approximates the convex hull.
        3. Refines this initial quadrilateral to ensure it fully circumscribes the convex hull.

        Note: The input polygon should be of shape (N, 2), where N is the number of vertices.

        :param simplify_polygons_larger_than: int | None. If a number is specified, the method will make a
                        preliminar Douglas-Peucker simplification of the Convex Hull if it has more than
                        simplify_polygons_larger_than vertices. This will speed up the process, but may
                        lead to a sub-optimal quadrilateral approximation.
        :param start_simplification_epsilon: float. The initial simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).
        :param max_simplification_epsilon: float. The maximum simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).
        :param simplification_epsilon_increment: float. The increment in the simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).

        :return: A tuple containing four tuples, each of which has two float elements representing the (x, y)
                coordinates of the quadrilateral's vertices. The vertices are order clockwise.

        :raises AssertionError: If the input polygon does not have a shape of (N, 2).
        """
        self._initial_guess = self.__find_initial_quadrilateral(max_sides_to_simplify=simplify_polygons_larger_than,
                                                            start_simplification_epsilon=start_simplification_epsilon,
                                                            max_simplification_epsilon=max_simplification_epsilon,
                                                            simplification_epsilon_increment=simplification_epsilon_increment)
        # Generate line equations from the initial guess
        self._line_equations = self.__polygon_vertices_to_line_equations(self._initial_guess)
        self.fitted_quadrilateral = self.__finetune_guess()
        # self.expanded_quadrilateral = self.__expand_quadrilateral()
        # return self.fitted_quadrilateral
        # self.expanded_quadrilateral = self.__expand_quadrilateral(expansion_margin=expansion_margin)
        # return self.expanded_quadrilateral
        # Ensure that line equations are set up before expanding
        if self._line_equations is None:
            raise ValueError("Line equations were not properly initialized.")

        self.expanded_quadrilateral = self.__expand_quadrilateral(expansion_margin=expansion_margin)
        return self.expanded_quadrilateral


    def __find_initial_quadrilateral(self, max_sides_to_simplify: int | None = 10,
                                     start_simplification_epsilon: float = 0.1,
                                     max_simplification_epsilon: float = 0.5,
                                     simplification_epsilon_increment: float = 0.02,
                                     max_combinations: int = 300) -> Polygon:
        """
        Internal method to find the initial approximating quadrilateral based on the vertices of the Convex Hull.
        To find the initial quadrilateral, we iterate through all 4-vertex combinations of the Convex Hull vertices
        and find the one with the highest Intersection over Union (IoU) with the Convex Hull. It will ensure that
        it is the best possible quadrilateral approximation to the input polygon.
        :param max_sides_to_simplify: int|None. If a number is specified, the method will make a
                        preliminar Douglas-Peucker simplification of the Convex Hull if it has more than
                        max_sides_to_simplify vertices. This will speed up the process, but may
                        lead to a sub-optimal quadrilateral approximation.
        :param start_simplification_epsilon: float. The initial simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).
        :param max_simplification_epsilon: float. The maximum simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).
        :param simplification_epsilon_increment: float. The increment in the simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).

        :param max_combinations: int. The maximum number of combinations to try. If the number of combinations
                        is larger than this number, the method will only run random max_combinations combinations.

        :return: Polygon. A Shapely Polygon object representing the initial quadrilateral approximation.
        """
        best_iou, best_quadrilateral = 0., None  # Variable to store the vertices of the best quadrilateral
        convex_hull_area = self.convex_hull_polygon.area

        # Simplify the Convex Hull if it has more than simplify_polygons_larger_than vertices
        simplified_polygon = self.__simplify_polygon(polygon=self.convex_hull_polygon,
                                                     max_sides=max_sides_to_simplify,
                                                     initial_epsilon=start_simplification_epsilon,
                                                     max_epsilon=max_simplification_epsilon,
                                                     epsilon_increment=simplification_epsilon_increment)

        all_combinations = tuple(combinations(mapping(simplified_polygon)['coordinates'][0], 4))

        # Limit the number of combinations to max_combinations if it's too large, to speed up the process
        if len(all_combinations) > max_combinations:
            all_combinations = sample(all_combinations, max_combinations)

        # Iterate through 4-vertex combinations to form potential quadrilaterals
        for vertices_combination in all_combinations:
            current_quadrilateral = Polygon(vertices_combination)
            assert current_quadrilateral.is_valid, f"Quadrilaterals generated from an ordered Convex Hull should be " \
                                                       f"always valid."

            # Calculate the Intersection over Union (IoU) between the Convex Hull and the current quadrilateral
            iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=current_quadrilateral,
                             precomputed_polygon_1_area=convex_hull_area)

            if iou > best_iou:
                best_iou, best_quadrilateral = iou, current_quadrilateral
                if iou >= 1.:
                    assert iou == 1., f"IoU should never be > 1.0. Got{iou}"
                    break  # We found the best possible quadrilateral, so we can stop iterating

        assert best_quadrilateral is not None, "No quadrilateral was found. This should never happen."

        return best_quadrilateral

    def __finetune_guess(self) -> Polygon:
        """
        Internal method to finetune the initial quadrilateral approximation to adjust to the input polygon.
        The method works by deciding which point of the initial polygon belongs to which side of the input polygon
        and fitting a line to each side of the input polygon. The intersection points between the lines will
        be the vertices of the new quadrilateral.

        :return: Polygon. A Shapely Polygon object representing the finetuned quadrilateral.
        """

        # initial_line_equations = self.__polygon_vertices_to_line_equations(polygon=self._initial_guess)
        # # Calculate the distance between each vertex of the input polygon and each line of the quadrilateral
        # distances = np.array(
        #     [line.distances_from_points(points=self._polygon_coords) for line in initial_line_equations],
        #     dtype=np.float32)
        # # For each point, get the index of the closest line
        # points_line_idx = np.argmin(distances, axis=0)

        # self._line_equations = tuple(
        #     self.__linear_regression(self._polygon_coords[points_line_idx == i], initial_guess=initial_guess)
        #         for i, initial_guess in enumerate(initial_line_equations)
        # )

        # new_quadrilateral_vertices = self.__find_polygon_vertices_from_lines(line_equations=self._line_equations)
        # return new_quadrilateral_vertices
        # Get the initial quadrilateral vertices
        initial_vertices = np.array(self._initial_guess.exterior.coords[:-1], dtype=np.float64).flatten()
        
        # Define the objective function (negative IoU)
        def objective(x):
            vertices = x.reshape((4, 2))
            quad = Polygon(vertices)
            if not quad.is_valid:
                return np.inf  # Penalize invalid quadrilaterals
            intersection_area = quad.intersection(self.convex_hull_polygon).area
            union_area = quad.union(self.convex_hull_polygon).area
            iou = intersection_area / union_area if union_area != 0 else 0
            return -iou  # Negative IoU for minimization

        # # Define convexity constraints
        # def convexity_constraint(x):
        #     vertices = x.reshape((4, 2))
        #     quad = Polygon(vertices)
        #     return 1 if quad.is_convex else -1  # Constraint satisfied if convex
        def convexity_constraint(x):
            vertices = x.reshape((4, 2))
            quad = Polygon(vertices)
            if not quad.is_valid:
                return -1  # Penalize invalid polygons to ensure we only deal with valid ones.
            # Check if the polygon is convex by comparing it to its convex hull.
            return 1 if quad.equals(quad.convex_hull) else -1

        constraints = {'type': 'ineq', 'fun': convexity_constraint}

        # Optimize vertex positions
        result = minimize(
            objective,
            initial_vertices,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        # Check if optimization was successful
        if not result.success:
            print("Optimization failed:", result.message)
            return self._initial_guess

        # Get optimized vertices
        optimized_vertices = result.x.reshape((4, 2))
        return Polygon(optimized_vertices)

    # def __linear_regression(self, points: np.ndarray, initial_guess: _Line = None) -> _Line:
    #     """
    #     Internal method that fits a line from a set of points using linear regression.
    #     :param points: np.ndarray. A numpy array of shape (N, 2) representing the points to fit the line to. Format X,Y
    #     :param initial_guess: _Line. An initial guess for the line equation. If None, the method will use the
    #                     linear regression method to find the best possible line.
    #     :return: _Line. A _Line object representing the fitted line.
    #     """

    #     def perpendicular_distance(params, points: np.ndarray):
    #         a, b, c = params
    #         x, y = points[:, 0], points[:, 1]
    #         return np.sum(np.abs(a * x + b * y + c)) / np.sqrt(a * a + b * b)

    #     if initial_guess is None:
    #         initial_guess = (1., -1., 0.)
    #     else:
    #         initial_guess = (initial_guess.A, initial_guess.B, initial_guess.C)

    #     result = minimize(perpendicular_distance, initial_guess, args=(points,), method='Nelder-Mead')
    #     A, B, C = result.x

    #     centroid = self.centroid
    #     return _Line(A=A, B=B, C=C, centroid=centroid)
    from scipy.optimize import minimize

    def __linear_regression(self, points: np.ndarray, initial_guess: _Line = None, 
                            method: str = 'Powell', maxiter: int = 10000, 
                            ftol: float = 1e-9, regularization: float = 0.0) -> _Line:
        """
        Internal method that fits a line from a set of points using customizable linear regression.
        
        :param points: np.ndarray. Array of shape (N, 2) with points to fit the line to (X, Y format).
        :param initial_guess: _Line. Initial guess for the line equation; default is None.
        :param method: str. Optimization method to use in minimize (default is 'Nelder-Mead').
        :param maxiter: int. Maximum iterations allowed for optimization (default is 1000).
        :param ftol: float. Tolerance for optimization (default is 1e-9 for higher precision).
        :param regularization: float. Regularization term to control line slope (default is 0 for no regularization).
        
        :return: _Line. A fitted line as a _Line object.
        """
        
        def perpendicular_distance(params, points: np.ndarray):
            a, b, c = params
            x, y = points[:, 0], points[:, 1]
            # Calculate total distance with optional regularization term
            distance = np.sum(np.abs(a * x + b * y + c)) / np.sqrt(a * a + b * b)
            return distance + regularization * (a**2 + b**2)

        if initial_guess is None:
            initial_guess = (1., -1., 0.)
        else:
            initial_guess = (initial_guess.A, initial_guess.B, initial_guess.C)

        result = minimize(
            perpendicular_distance, 
            initial_guess, 
            args=(points,), 
            method=method,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        
        A, B, C = result.x
        centroid = self.centroid  # Use centroid from the class to define _Line

        return _Line(A=A, B=B, C=C, centroid=centroid)

    def __expand_quadrilateral(self, expansion_margin: float = 0.0) -> Polygon:
        """
        Internal method that expands the initial quadrilateral approximation to make sure it contains all the vertices
        of the input polygon Convex Hull.
        Method:
            1. Move each line in their orthogonal direction (outwards) until it contains (or intersects)
               all the points of the Convex Hull in its inward direction
            2. Find the intersection points between the lines to calculate the vertices of the
               new expanded quadrilateral

        :param quadrilateral: Polygon. A Shapely Polygon object representing the initial quadrilateral approximation.

        :return: Polygon. A Shapely Polygon object representing the expanded quadrilateral.
        """
        # 1. Move each line in their orthogonal direction (outwards) until it contains (or intersects)
        #    all the points of the Convex Hull in its inward direction
        line_equations = tuple(line.copy() for line in self._line_equations)
        for line in line_equations:
            self.__move_line_to_contain_all_points(line=line, polygon=self.convex_hull_polygon)
            if expansion_margin > 0.0:
                line.move_line_outwards(distance=expansion_margin)
        # 3. Find the intersection points between the lines to calculate the vertices of the
        #    new expanded quadrilateral
        new_quadrilateral_vertices = self.__find_polygon_vertices_from_lines(line_equations=line_equations)
        return new_quadrilateral_vertices
    
    def get_expanded_quadrilateral(self) -> Polygon:
        """
        Public method that calls the private __expand_quadrilateral to expand the quadrilateral.
        """
        return self.__expand_quadrilateral()


    def __find_polygon_vertices_from_lines(self, line_equations: tuple[_Line]) -> tuple[tuple[float, float], ...]:
        """
        Internal method to calculate the vertices of a polygon from a tuple of line equations.

        :param line_equations: tuple[_Line]. A tuple of _Line objects representing the sides of the polygon.
        :return: tuple[tuple[float, float], ...]. A tuple of tuples representing the vertices of the polygon.
        """
        # Find the intersection between each line and its next one
        points = tuple(line1.get_intersection(other_line=line_equations[(i + 1) % len(line_equations)])
                     for i, line1 in enumerate(line_equations))
        # Order points clockwise
        points = self.__order_points_clockwise(pts=points)
        return points

    def __order_points_clockwise(self, pts: np.ndarray | tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:

        as_np = isinstance(pts, np.ndarray)
        if not as_np:
            pts = np.array(pts, dtype=np.float32)
        # Calculate the center of the points
        center = np.mean(pts, axis=0)

        # Compute the angles from the center
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

        # Sort the points by the angles in ascending order
        sorted_pts = pts[np.argsort(angles)]

        if not as_np:
            sorted_pts = tuple(tuple(pt) for pt in sorted_pts)
        return sorted_pts

    def __polygon_vertices_to_line_equations(self, polygon: Polygon) -> tuple[_Line]:
        """
        Converts a Polygon to a tuple of _Line objects representing the polygon's sides. Each line
        is associated with the polygon's centroid to support operations based on the polygon's geometric center.

        :param polygon: Polygon. A Shapely Polygon object to be converted into line equations.
        :return: tuple[_Line]. A tuple of _Line objects representing the sides of the polygon.
        """
        assert isinstance(polygon, Polygon), f"Expected a Shapely Polygon, got {type(polygon)} instead."
        if polygon.is_empty:
            return tuple()
        coords = polygon.exterior.coords
        centroid = self.centroid  # Retrieve centroid only once for efficiency
        return tuple(_Line(x1=x1, y1=y1, x2=x2, y2=y2, centroid=centroid)
                    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]))

    def __move_line_to_contain_all_points(self, line: _Line, polygon: Polygon) -> bool:
        """
        Internal method to move a line until it contains all points in the Convex Hull.

        :param line: _Line. The line to be moved.
        :param polygon: Polygon. The polygon to be contained by the line after moving it.

        :return: bool. True if the line was moved, False otherwise.
        """
        centroid = polygon.centroid
        centroid_sign = self.__sign(x=line.point_line_position(x=centroid.x, y=centroid.y))
        assert centroid_sign != 0, "The centroid of the polygon should never be on the line."

        max_distance, best_point = 0., None

        for (x, y) in self.convex_hull_polygon.exterior.coords[:-1]:
            point_position = line.point_line_position(x=x, y=y)
            if self.__sign(x=point_position) != centroid_sign:
                distance = line.distance_from_point(x=x, y=y)
                if distance > max_distance:
                    max_distance, best_point = distance, (x, y)

        if best_point is not None:
            x, y = best_point
            line.move_line_to_intersect_point(x=x, y=y)
            return True
        return False


    # -------------------------------- HELPER METHODS -------------------------------- #

    def __simplify_polygon(self, polygon: Polygon|GeometryCollection, max_sides: int | None,
                           initial_epsilon: float = 0.1, max_epsilon: float = 0.5,
                           epsilon_increment: float = 0.02, iou_threshold: float = 0.8):
        """
        Internal method to simplify a polygon using the Douglas-Peucker algorithm.
        :param polygon: Polygon or GeometryCollection. The polygon or collection of geometries to simplify.
        :param max_sides: int|None. The maximum number of sides the polygon can have after simplification.
        If None, no simplification will be performed.
        :param max_epsilon: float. The maximum tolerance value for the Douglas-Peucker algorithm.
        :param initial_epsilon: float. The initial tolerance value for the Douglas-Peucker algorithm.
        :param epsilon_increment: float. The incremental step for the tolerance value.

         :return: Polygon. The simplified polygon.
         """

        if isinstance(polygon, Polygon):
            polygon_to_simplify = polygon
        elif isinstance(polygon, GeometryCollection):
            # Find the first Polygon in the collection (assuming there's only one)
            polygon_to_simplify = next((geom for geom in polygon.geoms if isinstance(geom, Polygon)), None)
            if polygon_to_simplify is None:
                raise ValueError("No Polygon found in GeometryCollection.")
        else:
            raise TypeError("Expected Polygon or GeometryCollection, got {type(polygon)}.")

        # Now simplify the polygon_to_simplify
        if polygon_to_simplify is None or max_sides is None or len(polygon_to_simplify.exterior.coords) - 1 <= max_sides:
            return polygon_to_simplify  # No simplification needed

        assert max_epsilon > 0., f"max_epsilon should be a float greater than 0. Got {max_epsilon}."
        assert initial_epsilon > 0., f"initial_epsilon should be a float greater than 0. Got {initial_epsilon}."
        assert epsilon_increment > 0., f"epsilon_increment should be a float greater than 0. Got {epsilon_increment}."

        simplified_polygon = polygon_to_simplify
        original_polygon_area = polygon_to_simplify.area

        epsilon = initial_epsilon
        while epsilon <= max_epsilon:
            # Simplify the polygon
            simplified_polygon_unconfirmed = simplified_polygon.simplify(epsilon, preserve_topology=True)
            n_sides = len(simplified_polygon_unconfirmed.exterior.coords) - 1
            # If the polygon has less than 4 sides, it becomes invalid, get the previous one
            if n_sides < 4:
                break
            # If the polygon has less than max_sides, we have it. Return it or the previous one depending on the IoU
            elif len(simplified_polygon_unconfirmed.exterior.coords) - 1 <= max_sides:
                iou = self.__iou(polygon1=simplified_polygon_unconfirmed, polygon2=self.convex_hull_polygon,
                                 precomputed_polygon_1_area=original_polygon_area)
                # If the IoU is beyond the threshold, we accept the polygon. Otherwise, return the previous one
                if iou > iou_threshold:
                    # We accept the polygon
                    simplified_polygon = simplified_polygon_unconfirmed
                return simplified_polygon
            else:
                # If the polygon has more than max_sides, that's our best guess so far, but keep trying
                simplified_polygon = simplified_polygon_unconfirmed
                epsilon += epsilon_increment

        return simplified_polygon

    def __iou(self, polygon1: Polygon, polygon2: Polygon, precomputed_polygon_1_area: float | None = None) -> float:
        """
        Calculate the Intersection over Union (IoU) between two polygons.

        :param polygon1: Polygon. The first polygon.
        :param polygon2: Polygon. The second polygon.
        :param precomputed_polygon_1_area: float|None. The area of the first polygon. If None, it will be computed.
        :return: float. The IoU value.
        """
        if precomputed_polygon_1_area is None:
            precomputed_polygon_1_area = polygon1.area
        # Calculate the intersection and union areas
        intersection = polygon1.intersection(polygon2).area
        union = precomputed_polygon_1_area + polygon2.area - intersection
        # Return the IoU value
        return (intersection / union) if union != 0. else 0.

    @staticmethod
    def __sign(x: int | float) -> int:
        """
        Return the sign of a number.
        :param x: float. The number to check.
        :return: int. 1 if x > 0, -1 if x < 0, 0 if x == 0.
        """
        return 1 if x > 0 else (-1 if x < 0 else 0)

    # def plot(self):
    #     """
    #     Plot the convex hull and the best-fitting quadrilateral for debugging purposes.
    #     This function imports matplotlib.pyplot locally, so the library is not required for the entire class.
    #     """
    #     try:
    #         import matplotlib.pyplot as plt
    #     except ImportError:
    #         raise ImportError("This function requires matplotlib to be installed. Please install it first.")

    #     # Plot the original polygon as a set of alpha 0.4 points
    #     plt.plot(self._polygon_coords[:, 0], self._polygon_coords[:, 1], alpha=0.3,  linestyle='-', marker='o', label='Input Polygon')

    #     # Plot the convex hull as a filled polygon
    #     x, y = self.convex_hull_polygon.exterior.xy
    #     plt.fill(x, y, alpha=0.4, label='Convex Hull', color='orange')

    #     # Plot the initial quadrilateral if it exists as a semi-transparent dashed line
    #     if self._initial_guess is not None:
    #         # Calculate the IoU between the Convex Hull and the best-fitting quadrilateral
    #         iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=Polygon(self._initial_guess))
    #         x, y = self._initial_guess.exterior.xy
    #         plt.plot(x, y, linestyle='--', alpha=0.5, color='green', label=f'Initial Guess (IoU={iou:.3f})')

    #     # Plot the best quadrilateral if it exists
    #     if self.fitted_quadrilateral is not None:
    #         # Calculate the IoU between the Convex Hull and the best-fitting quadrilateral
    #         iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=Polygon(self.fitted_quadrilateral))
    #         x, y = zip(*self.fitted_quadrilateral)
    #         plt.plot(x + (x[0],), y + (y[0],), label=f'Fitted Quadrilateral (IoU={iou:.3f})')

    #         # Mark the corners of the best quadrilateral with 'X'
    #         plt.scatter(x, y, marker='x', color='red')

    #     plt.axis('equal')
    #     plt.xlabel('X')
    #     # Reverse Y axis
    #     plt.ylabel('Y')
    #     plt.title('Quadrilateral Fitting')
    #     plt.legend()
    #     ax = plt.gca()  # Get current axes
    #     ax.invert_yaxis()
    #     plt.grid(True)
    #     plt.show()

    def plot(self):
        """
        Plot the convex hull and the best-fitting quadrilateral for debugging purposes.
        This function imports matplotlib.pyplot locally, so the library is not required for the entire class.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("This function requires matplotlib to be installed. Please install it first.")

        # Plot the original polygon as a set of alpha 0.4 points
        plt.plot(self._polygon_coords[:, 0], self._polygon_coords[:, 1], alpha=0.3, linestyle='-', marker='o', label='Input Polygon')

        # Plot the convex hull as a filled polygon
        x, y = self.convex_hull_polygon.exterior.xy
        plt.fill(x, y, alpha=0.4, label='Convex Hull', color='orange')

        # Plot the initial quadrilateral if it exists as a semi-transparent dashed line
        if self._initial_guess is not None:
            # Calculate the IoU between the Convex Hull and the best-fitting quadrilateral
            iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=self._initial_guess)
            x, y = self._initial_guess.exterior.xy
            plt.plot(x, y, linestyle='--', alpha=0.5, color='green', label=f'Initial Guess (IoU={iou:.3f})')

        # Plot the best quadrilateral if it exists
        if self.fitted_quadrilateral is not None:
            # Calculate the IoU between the Convex Hull and the best-fitting quadrilateral
            iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=self.fitted_quadrilateral)
            x, y = self.fitted_quadrilateral.exterior.xy
            plt.plot(list(x) + [x[0]], list(y) + [y[0]], label=f'Fitted Quadrilateral (IoU={iou:.3f})', color='blue', linestyle='-')
            print(f'Four corners of the fitted quadrilateral: {list(zip(x, y))}')

            # Mark the corners of the best quadrilateral with 'X'
            plt.scatter(x, y, marker='x', color='red')

        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().invert_yaxis()  # Reverse Y axis
        plt.title('Quadrilateral Fitting')
        plt.legend()
        plt.grid(True)
        plt.show()
