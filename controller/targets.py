from collections import deque
import numpy as np

class Target():
    def __init__(self, id, label, threshold=0.3, len_history=10, min_distance=1.5):
        self.id = id          # class ID
        self.label = label    # Descriptive label for the target (e.g., 'stop_sign')
        self.threshold = threshold  # Minimum ratio of positive detections to consider the target visible
        self.len_history = len_history  # Number of recent detection results to track
        self.history = deque([False] * self.len_history, maxlen=self.len_history)  # Detection history buffer
        self.has_reacted = False  # Flag to track if the system has already reacted to this target
        self.min_distance = min_distance  # Distance threshold to trigger a reaction (e.g., braking), has to be at least this close
        self.position: np.ndarray | None = None # Latest detected position of the target 

    @property
    def visible(self):
        # Compute visibility based on the average detection history
        return np.mean(self.history) > self.threshold

    @property
    def in_range(self):
        # Return True if position is known and x-distance is within the threshold
        return self.position is not None and self.position[0] <= self.min_distance


    def update(self, position, node):
        self.position = position

        # Update detection history based on current observation
        if np.isnan(position).any():
            # Object NOT detected this frame
            self.history.append(False)
        else:
            # Object IS detected this frame
            self.history.append(True)

        # Trigger action if target is visible and in range and not yet reacted
        if self.visible and not self.has_reacted:
            # node.get_logger().info(f"target={self.label}, in update (target visible and not reacted)")  # print debugging
            if self.in_range:
                node.get_logger().info(f"target={self.label}, in update (IS in_range, about to react), pos={self.position}")  # print debugging
                self.has_reacted = self.react(node)  # Pass node here
            else:
                try:
                    node.get_logger().info(f'target={self.label}, in update (NOT in range yet at pos={self.position})') # print debugging
                except:
                    node.get_logger().info(f'target={self.label}, in update (NOT in range yet, BAD distance)') # print debugging
                
        # Reset if target is no longer visible
        elif not any(self.history) and self.has_reacted:
            # node.get_logger().info(f"in update (target not visible and has reacted) for target:{self}")  # print debugging
            # node.get_logger().info(f"target={self.label}, in update (no true hist & has react) (RESET react)")  # print debugging
            node.get_logger().info(f"target={self.label}, in update, RESET react (ready for another detection)")  # print debugging
            self.has_reacted = False

    def react(self, node) -> bool:
        # Override this method in subclasses
        return False
    
    def __repr__(self):
        # Can override this method in subclasses
        return f'<Target with id={self.id}, label={self.label}>'

