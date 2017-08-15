/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static constexpr int kNumParticles = 100;
static constexpr double kZeroYawRate = 0.00001;

static default_random_engine gen;

namespace {
bool is_yaw_rate_equals_zero(double yaw_rate) {
  return fabs(yaw_rate) < kZeroYawRate;
}

bool in_range(double sensor_range, const Particle& p,
              const Map::single_landmark_s& landmark) {
  return dist(p.x, p.y, landmark.x_f, landmark.y_f) < sensor_range;
}
}  // namespace

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  normal_distribution<double> d_x(0, std[0]);
  normal_distribution<double> d_y(0, std[1]);
  normal_distribution<double> d_theta(0, std[2]);

  num_particles = kNumParticles;
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = x + d_x(gen);
    p.y = y + d_y(gen);
    p.theta = theta + d_theta(gen);
    p.weight = 1.0;
    particles.emplace_back(std::move(p));
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> d_x(0, std_pos[0]);
  normal_distribution<double> d_y(0, std_pos[1]);
  normal_distribution<double> d_theta(0, std_pos[2]);

  for (auto& particle : particles) {
    if (is_yaw_rate_equals_zero(yaw_rate)) {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      auto p_theta = particle.theta + yaw_rate * delta_t;
      particle.x += velocity / yaw_rate * (sin(p_theta) - sin(particle.theta));
      particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(p_theta));
      particle.theta = p_theta;
    }
    particle.x += d_x(gen);
    particle.y += d_y(gen);
    particle.theta += d_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for (auto& o : observations) {
    double min_dist = numeric_limits<double>::max();
    int map_id = -1;
    for (const auto& p : predicted) {
      double c_dist = dist(o.x, o.y, p.x, p.y);
      if (c_dist < min_dist) {
        min_dist = c_dist;
        map_id = p.id;
      }
    }
    // set the observation's id to the nearest predicted landmark's id
    o.id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  double dev_x = std_landmark[0];
  double dev_y = std_landmark[1];

  for (auto& particle : particles) {
    // Populate the list of the landmark predictions based on the location
    // of the particle.
    vector<LandmarkObs> predictions;
    predictions.reserve(map_landmarks.landmark_list.size());
    for (const auto& s_landmark : map_landmarks.landmark_list) {
      if (in_range(sensor_range, particle, s_landmark)) {
        predictions.push_back(LandmarkObs{s_landmark.id_i, s_landmark.x_f,
            s_landmark.y_f});
      }
    }
    // Transform observations from particle into the map coordinates.
    vector<LandmarkObs> m_observations;
    m_observations.reserve(observations.size());
    std::transform(
        observations.begin(), observations.end(), back_inserter(m_observations),
        [&particle](const LandmarkObs& o) {
          double m_x = cos(particle.theta)*o.x - sin(particle.theta)*o.y +
          particle.x;
          double m_y = sin(particle.theta)*o.x + cos(particle.theta)*o.y +
          particle.y;
          return LandmarkObs {o.id, m_x, m_y};
        });
    // Now when we have predicted and observed landmarks, we may try to associate
    // those.
    dataAssociation(predictions, m_observations);

    // Finally, update the weight of the particle.
    double weight = 1.0;
    for (const auto& o : m_observations) {
      for (const auto& p : predictions) {
        if (p.id != o.id) {
          continue;
        }
        double observation_w = 1.0 / (2.0 * M_PI * dev_x * dev_y)
            * exp(
                -1.
                    * (pow(p.x - o.x, 2) / (2 * pow(dev_x, 2))
                        + pow(p.y - o.y, 2) / (2 * pow(dev_y, 2))));
        weight *= observation_w;
        break;
      }
    }
    particle.weight = weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> new_particles;
  new_particles.reserve(particles.size());

  vector<double> weights;
  weights.reserve(particles.size());
  std::transform(particles.begin(), particles.end(), back_inserter(weights),
                 [](const Particle& p) {return p.weight;});

  uniform_int_distribution<int> index_dist(0, num_particles - 1);
  auto index = index_dist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> weight_dist(0.0, max_weight);

  double beta = 0.0;
  for (int i = 0; i < num_particles; i++) {
    beta += weight_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
