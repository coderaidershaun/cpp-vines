#pragma once

#include <string_view>


enum class SmartError {
  ArrayLengthZero,
  ArraysLengthMismatch,
  CsvFailedToOpen,
  CsvIncorrectColumnIndex,
  CsvOther,
  CopulaNotFitted,
  CopulaFitsFailed,
  MissingWeighting,
  DivisionByZero,
  PriceZero,
  PriceNegative,
  PriceInfinite,
  PriceNan
};

constexpr std::string_view error_to_string(SmartError error) {
  switch(error) {
    case SmartError::ArrayLengthZero:
      return "ArrayLengthZero: An array without any length was passed";
    case SmartError::ArraysLengthMismatch:
      return "ArraysLengthMismatch: Both arrays should have the same length";
    case SmartError::CsvFailedToOpen:
      return "CsvFailure: Failed to open csv";
    case SmartError::CsvIncorrectColumnIndex:
      return "CsvIncorrectColumnIndex: Column index must be greater than zero";
    case SmartError::CsvOther:
      return "CsvOther: An error occured parsing the csv data";
    case SmartError::CopulaNotFitted:
      return "CopulaNotFitted: Copula fitting has not been run";
    case SmartError::CopulaFitsFailed:
      return "CopulaFitsFailed: All copula fits failed";
    case SmartError::DivisionByZero:
      return "DivisionByZero: There was a division by zero attempted";
    case SmartError::MissingWeighting:
      return "MissingWeighting: Edge weightings must be set before calling this function";
    case SmartError::PriceZero:
      return "PriceZero: prices cannot be zero";
    case SmartError::PriceNegative:
      return "PriceZero: prices must be greater than zero";
    case SmartError::PriceInfinite:
      return "PriceZero: prices cannot be infinite";
    case SmartError::PriceNan:
      return "PriceZero: prices cannot be NaN";
  }

  return "Error: Unknown SmartError occured";
}
