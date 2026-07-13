#pragma once

//! Allows for importing of provided prices.csv for testing.

#include <exception>
#include <expected>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <errors.hpp>


std::expected<std::vector<double>, SmartError> read_csv(
  const std::string& filename,
  int column_index
) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return std::unexpected(SmartError::CsvFailedToOpen);
  }

  if (column_index < 0) {
    return std::unexpected(SmartError::CsvIncorrectColumnIndex);
  }

  std::vector<double> values;
  std::string line;

  std::getline(file, line);

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    int current_column = 0;
    bool found_column = false;

    while (std::getline(ss, cell, ',')) {
      if (current_column == column_index) {
        if (cell.empty()) {
          return values;
        }

        try {
          double value = std::stod(cell);
          values.push_back(value);
          found_column = true;
          break;
        } catch (const std::exception&) {
          return std::unexpected(SmartError::CsvOther);
        }
      }

      current_column += 1;
    }

    if (!found_column) {
      return values;
    }
  }

  return values;
}
