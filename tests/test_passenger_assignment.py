"""Capacity and demand conservation on small, hand-calculated networks."""

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transit_rl.config import ExperimentConfig
from transit_rl.data.network import build_network
from transit_rl.environment.passenger_assignment import PassengerAssignment
from transit_rl.environment.service_plan import RouteService


class CapacityAssignmentTests(unittest.TestCase):
    def assign(self, n, links, demands, plan, **overrides):
        cfg = dataclasses.replace(ExperimentConfig(), num_stops=n, num_hours=1,
                                  od_start_file_hour=0, max_route_length=n, **overrides)
        cfg.validate()
        network = build_network(links, n, True)
        od = np.zeros((n, n))
        for i, j, demand in demands:
            od[i - 1, j - 1] = demand
        before = od.copy()
        od.setflags(write=False)
        assigner = PassengerAssignment(network, cfg, [od])
        result = assigner.assign(1, plan)

        # Every OD passenger is either boarded on a complete trip or unmet.
        expected = before.copy()
        if cfg.ignore_od_diagonal:
            np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(result.assigned_demand + result.unserved_demand, expected)
        np.testing.assert_allclose(result.direct_demand + result.transfer_demand, result.assigned_demand)
        np.testing.assert_array_equal(od, before)
        self.assertAlmostEqual(result.served_passengers, result.assigned_demand.sum())
        self.assertAlmostEqual(result.unserved_passengers, result.unserved_demand.sum())
        self.assertAlmostEqual(result.served_passengers + result.unserved_passengers, result.total_demand)
        self.assertAlmostEqual(result.direct_passengers + result.transfer_passengers, result.served_passengers)
        self.assertAlmostEqual(result.overload_passengers, 0)
        for route, peak in zip(plan, result.route_peak_load):
            self.assertLessEqual(peak, route.frequency * cfg.bus_capacity * cfg.bus_load_factor + 1e-9)
        for array in (result.assigned_demand, result.unserved_demand,
                      result.direct_demand, result.transfer_demand):
            self.assertFalse(array.flags.writeable)
        self.assertIs(assigner.assign(1, plan), result)
        return result

    def test_fifty_passengers_per_departure_and_frequency_scaling(self):
        for frequency, served, unmet in ((1, 50, 30), (2, 80, 0)):
            with self.subTest(frequency=frequency):
                result = self.assign(2, [(1, 2, 5)], [(1, 2, 80)],
                                     (RouteService((1, 2), frequency),))
                self.assertEqual(result.served_passengers, served)
                self.assertEqual(result.unserved_passengers, unmet)
                self.assertEqual(result.total_in_vehicle_time, served * 5)
                self.assertEqual(result.total_waiting_time, served * 30 / frequency)
                self.assertEqual(result.average_travel_time, 5 + 30 / frequency)

    def test_load_factor_changes_boarding_limit(self):
        for factor, served in ((1, 40), (1.25, 50)):
            with self.subTest(factor=factor):
                result = self.assign(2, [(1, 2, 5)], [(1, 2, 80)],
                                     (RouteService((1, 2), 1),), bus_load_factor=factor)
                self.assertEqual(result.served_passengers, served)

    def test_shared_segments_compete_for_capacity(self):
        result = self.assign(3, [(1, 2, 5), (2, 3, 5)],
                             [(1, 2, 30), (1, 3, 40), (2, 3, 30)],
                             (RouteService((1, 2, 3), 1),))
        self.assertEqual(result.served_passengers, 80)
        self.assertEqual(result.assigned_demand[0, 2], 20)
        self.assertEqual(result.unserved_demand[0, 2], 20)
        self.assertEqual(result.route_peak_load, (50,))

    def test_seats_reused_after_alighting(self):
        result = self.assign(3, [(1, 2, 5), (2, 3, 5)],
                             [(1, 2, 50), (2, 3, 50)],
                             (RouteService((1, 2, 3), 1),))
        self.assertEqual(result.served_passengers, 100)
        self.assertEqual(result.route_peak_load, (50,))

    def test_directions_have_independent_capacity(self):
        result = self.assign(2, [(1, 2, 5)], [(1, 2, 80), (2, 1, 80)],
                             (RouteService((1, 2), 1),))
        self.assertEqual(result.served_passengers, 100)
        self.assertEqual(result.unserved_passengers, 60)
        result = self.assign(2, [(1, 2, 5)], [(1, 2, 80), (2, 1, 80)],
                             (RouteService((1, 2), 1),), routes_are_bidirectional=False)
        self.assertEqual(result.served_passengers, 50)
        self.assertEqual(result.unserved_demand[1, 0], 80)

    def test_common_lines_rebalance_after_one_fills(self):
        result = self.assign(3, [(1, 2, 2), (2, 3, 2), (1, 3, 4)],
                             [(1, 2, 40), (1, 3, 120)],
                             (RouteService((1, 2, 3), 1), RouteService((1, 3), 2)))
        self.assertEqual(result.assigned_demand[0, 2], 110)
        self.assertEqual(result.unserved_demand[0, 2], 10)
        self.assertEqual(result.route_peak_load, (50, 100))
        # First 30 split 10/20 at combined f=3; last 80 use f=2.
        self.assertEqual(result.total_waiting_time, 40 * 30 + 30 * 10 + 80 * 15)

    def test_slower_direct_path_is_used_after_fast_route_fills(self):
        result = self.assign(3, [(1, 2, 5), (1, 3, 5), (3, 2, 5)],
                             [(1, 2, 80)],
                             (RouteService((1, 2), 1), RouteService((1, 3, 2), 1)))
        self.assertEqual(result.direct_passengers, 80)
        self.assertEqual(result.route_peak_load, (50, 30))
        self.assertEqual(result.total_in_vehicle_time, 50 * 5 + 30 * 10)

    def test_transfer_requires_capacity_on_every_leg(self):
        result = self.assign(3, [(1, 2, 5), (2, 3, 7)], [(1, 3, 80)],
                             (RouteService((1, 2), 2), RouteService((2, 3), 1)))
        self.assertEqual(result.one_transfer_passengers, 50)
        self.assertEqual(result.unserved_passengers, 30)
        self.assertEqual(result.route_peak_load, (50, 50))
        self.assertEqual(result.total_in_vehicle_time, 50 * 12)
        self.assertEqual(result.total_waiting_time, 50 * 45)
        self.assertEqual(result.total_transfer_penalty_time, 50 * 5)

    def test_rejected_transfer_does_not_consume_upstream_capacity(self):
        result = self.assign(5, [(1, 3, 5), (3, 4, 5), (2, 3, 5), (3, 5, 5)],
                             [(1, 4, 50), (2, 4, 50), (2, 5, 50)],
                             (RouteService((1, 3), 1), RouteService((3, 4), 1),
                              RouteService((2, 3, 5), 1)))
        self.assertEqual(result.assigned_demand[0, 3], 50)
        self.assertEqual(result.unserved_demand[1, 3], 50)
        # Rejected 2->4 passengers must leave 2->3 capacity for the 2->5 trip.
        self.assertEqual(result.assigned_demand[1, 4], 50)
        self.assertEqual(result.served_passengers, 100)
        self.assertEqual(result.route_peak_load, (50, 50, 50))

    def test_alternative_transfer_path_uses_remaining_capacity(self):
        for rule in ('min_transfers_first', 'min_generalised_time'):
            with self.subTest(rule=rule):
                result = self.assign(4, [(1, 2, 2), (2, 4, 2), (1, 3, 3), (3, 4, 3)],
                                     [(1, 4, 80)],
                                     (RouteService((1, 2), 1), RouteService((2, 4), 1),
                                      RouteService((1, 3), 1), RouteService((3, 4), 1)),
                                     path_choice_rule=rule)
                self.assertEqual(result.one_transfer_passengers, 80)
                self.assertEqual(result.route_peak_load, (50, 50, 30, 30))

    def test_direct_overflow_can_use_transfer_path(self):
        result = self.assign(3, [(1, 3, 5), (1, 2, 5), (2, 3, 5)], [(1, 3, 80)],
                             (RouteService((1, 3), 1), RouteService((1, 2), 1),
                              RouteService((2, 3), 1)))
        self.assertEqual(result.direct_passengers, 50)
        self.assertEqual(result.one_transfer_passengers, 30)
        self.assertEqual(result.unserved_passengers, 0)

    def test_two_transfers_use_bottleneck_capacity(self):
        plan = (RouteService((1, 2), 2), RouteService((2, 3), 1), RouteService((3, 4), 2))
        result = self.assign(4, [(1, 2, 2), (2, 3, 2), (3, 4, 2)], [(1, 4, 80)], plan)
        self.assertEqual(result.two_transfer_passengers, 50)
        self.assertEqual(result.unserved_passengers, 30)
        self.assertEqual(result.route_peak_load, (50, 50, 50))
        self.assertEqual(result.total_transfer_penalty_time, 50 * 10)
        result = self.assign(4, [(1, 2, 2), (2, 3, 2), (3, 4, 2)], [(1, 4, 80)],
                             plan, enable_two_transfer_paths=False)
        self.assertEqual(result.served_passengers, 0)

    def test_rerouting_respects_trip_time_limit(self):
        result = self.assign(3, [(1, 2, 5), (1, 3, 10), (3, 2, 10)], [(1, 2, 80)],
                             (RouteService((1, 2), 1), RouteService((1, 3, 2), 1)),
                             max_acceptable_trip_minutes=40)
        self.assertEqual(result.served_passengers, 50)
        self.assertEqual(result.unserved_passengers, 30)
        self.assertEqual(result.route_peak_load, (50, 0))

    def test_empty_plan_zero_demand_and_diagonal(self):
        result = self.assign(2, [(1, 2, 5)], [(1, 2, 80), (1, 1, 20)], ())
        self.assertEqual(result.total_demand, 80)
        self.assertEqual(result.served_passengers, 0)
        self.assertEqual(result.unserved_passengers, 80)
        self.assertEqual(result.average_travel_time, 0)
        result = self.assign(2, [(1, 2, 5)], [], (RouteService((1, 2), 1),))
        self.assertEqual(result.total_demand, 0)
        self.assertEqual(result.average_waiting_time, 0)

    def test_invalid_load_factors_are_rejected(self):
        for factor in (0, -1, float('inf'), float('nan')):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                dataclasses.replace(ExperimentConfig(), bus_load_factor=factor).validate()


if __name__ == '__main__':
    unittest.main()
