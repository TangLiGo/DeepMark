import itertools


class GaloisField:

    def __init__(self):
        self.galois_field_alphabet = 256
        # x^8 + x^4 + x^3 + x^2 + 1
        self.primitive_polynomial = 285

    def integer_inverse(self, integer):
        """Inverse function, return X^-1. Using next: X^-1 = (X^m)-2"""
        return self.polynomial_power(integer, self.galois_field_alphabet - 2)

    def polynomials_division(self, dividend, divisor):
        """Fast polynomial division by using Extended Synthetic Division and optimized for GF(2^p) computations
        (doesn't work with standard polynomials outside of this galois field). """
        # CAUTION: this function expects polynomials to follow the opposite convention at decoding: the terms must go
        # from the biggest to lowest degree (while most other functions here expect a list from lowest to biggest
        # degree). eg: 1 + 2x + 5x^2 = [5, 2, 1], NOT [1, 2, 5]

        # Copy the dividend list and pad with 0 where the ecc bytes will be computed
        msg_out = bytearray(dividend)
        # normalizer = divisor[0] # precomputing for performance

        for i in range(len(dividend) - (len(divisor) - 1)):
            # msg_out[i] /= normalizer # for general polynomial division (when polynomials are non-monic), the usual
            # way of using synthetic division is to divide the divisor g(x) with its leading coefficient (call it a).
            # In this implementation, this means:we need to compute: coef = msg_out[i] / gen[0]. For more infos,
            # see http://en.wikipedia.org/wiki/Synthetic_division
            coef = msg_out[i]  # precaching

            if coef != 0:
                # in synthetic division, we always skip the first coefficient of the divisor,
                # because it's only used to normalize the dividend coefficient
                for j in range(1, len(divisor)):
                    if divisor[j] != 0:  # log(0) is undefined
                        # equivalent to the more mathematically correct (but xoring directly is faster):
                        # msg_out[i + j] += -divisor[j] * coef
                        msg_out[i + j] ^= self.integer_multiplication(divisor[j], coef)

        # The resulting msg_out contains both the quotient and the remainder,
        # the remainder being the size of the divisor (the remainder has necessarily the same degree as the divisor
        # -- not length but degree == length-1 -- since it's what we couldn't divide from the dividend),
        # so we compute the index where this separation is, and return the quotient and remainder.
        separator = -(len(divisor) - 1)
        return msg_out[:separator], msg_out[separator:]  # return quotient, remainder.

    def polynomials_multiplication(self, first_pol, second_pol):
        """Multiply two polynomials, inside Galois Field"""
        # Pre-allocate the result array
        result_pol = bytearray(len(first_pol) + len(second_pol) - 1)
        # Compute the polynomial multiplication (just like the outer product of two vectors, we multiply each
        # coefficients of p with all coefficients of q)
        for j in range(len(second_pol)):
            for i in range(len(first_pol)):
                # equivalent to: result_pol[i + j] = gf_add(result_pol[i+j], gf_mul(p[i], q[j]))
                # you can see it's your usual polynomial multiplication
                result_pol[i + j] ^= self.integer_multiplication(first_pol[i], second_pol[j])
        return result_pol

    def polynomial_power(self, pol, degree):
        """Exponentiation of a polynomial
        Cycle, that (degree - 2) times multiply pol by himself"""
        result = 1
        if degree == 0:
            return result
        elif degree > 0:
            for _ in range(0, degree):
                result = self.integer_multiplication(result, pol)
        elif degree < 0:
            for _ in range(0, degree, -1):
                result = self.integer_multiplication(result, pol)
            result = self.integer_inverse(result)
        return result

    def integer_multiplication(self, first_int, second_int):
        """Multiply two integers, inside Galois Field
        Galois Field integer multiplication using Russian Peasant Multiplication algorithm
        """
        result_int = 0
        # while second integer is above 0
        while second_int:
            # If second integer is odd, then add the corresponding first integer to result_int
            # (sum of all first integer's corresponding to odd second integer's will give the final result).
            if second_int & 1:
                result_int = result_int ^ first_int
            # Divide the number by two without a remainder
            second_int = second_int >> 1
            # Multiply twice the first integer
            first_int = first_int << 1
            if first_int & self.galois_field_alphabet:
                first_int = first_int ^ self.primitive_polynomial

        return result_int

    def polynomial_and_integer_multiplication(self, polynomial, integer):
        """Multiply a polynomial by a scalar (int), inside Galois Field
        Multiply each coefficient of the polynomial by a scalar
        """
        return bytearray([self.integer_multiplication(polynomial[i], integer) for i in range(len(polynomial))])

    @staticmethod
    def polynomial_sum(first_pol, second_pol):
        """Adds two polynomials, inside Galois Field"""
        result_pol = bytearray(max(len(first_pol), len(second_pol)))
        for i in range(0, len(first_pol)):
            result_pol[i + len(result_pol) - len(first_pol)] = first_pol[i]
        for i in range(0, len(second_pol)):
            result_pol[i + len(result_pol) - len(second_pol)] ^= second_pol[i]
        return result_pol


class ReedSolomonCoder(GaloisField):
    def __init__(self, redundant_characters=10):
        """The init method for ReedSolomon coder"""
        # Using GaloisField initial method
        super().__init__()
        # Primitive element of a Gracie's field
        self.primitive_element = 2
        # Number of redundant characters
        self.redundant_characters = redundant_characters

    def encode(self, message):
        """Reed-Solomon message encoder"""
        # Execute method, which return generator polynomial
        generator = self.calculate_polynomial_generator()
        # Divide a message by a Generating Polynomial
        _, remainder = self.polynomials_division(dividend=bytearray(message) + bytearray(len(generator) - 1),
                                                 divisor=generator)
        return bytearray(message + remainder)

    def calculate_polynomial_generator(self):
        """Calculation polynomial generator"""
        # Init array with byte 1, because the highest degree always does not have coefficient
        result_pol = bytearray([1])
        for i in range(self.redundant_characters):
            raised_to_the_power = self.polynomial_power(self.primitive_element, i)
            result_pol = self.polynomials_multiplication(result_pol, [1, raised_to_the_power])
        return result_pol

    def decode(self, message):
        """Reed-Solomon message decoder"""
        # Calculate polynomial of syndromes. If all syndromes coefficients are 0, then just return the codeword as-is.
        syndromes = self.calculate_polynomial_of_syndromes(message)
        if max(syndromes) == 0:
            return message
        # Calculate error locator
        errors_locator = self.calculate_error_locator(syndromes)
        # Reverse locator
        errors_locator = errors_locator[::-1]
        # Find errors_positions by locator
        errors_positions = self.find_errors_by_locator(errors_locator, len(message))
        # Correct errors_positions
        message = self.correct_errors(message, syndromes, errors_positions, errors_locator)
        return message

    def calculate_polynomial_of_syndromes(self, message):
        """Given the received codeword msg and the number of error correcting symbols (nsym), computes the syndromes
        polynomial. Mathematically, it's essentially equivalent to a Fourrier Transform (Chien search being the
        inverse).
        """
        # Note the "[0] +" : we add a 0 coefficient for the lowest degree (the constant). This effectively shifts the
        # syndromes, and will shift every computations depending on the syndromes (such as the errors locator
        # polynomial, errors evaluator polynomial, etc. but not the errors positions). This is not necessary as
        # anyway syndromes are defined such as there are only non-zero coefficients (the only 0 is the shift of the
        # constant here) and subsequent computations will/must account for the shift by skipping the first iteration
        # (eg, the often seen range(1, n-k+1)), but you can also avoid prepending the 0 coeff and adapt every
        # subsequent computations to start from 0 instead of 1.
        return [0] + [self.calculate_polynomial_by_x(message, self.polynomial_power(self.primitive_element, i)) for i in
                      range(self.redundant_characters)]

    def calculate_polynomial_by_x(self, poly, x):
        """Evaluates a polynomial in GF(2^p) given the value for x. This is based on Horner's scheme for maximum
        efficiency. """
        y = poly[0]
        for i in range(1, len(poly)):
            y = self.integer_multiplication(y, x) ^ poly[i]
        return y

    def calculate_error_locator(self, syndromes):
        """Find error/errata locator and evaluator polynomials with Berlekamp-Massey algorithm"""
        # The idea is that BM will iteratively estimate the error locator polynomial.
        # To do this, it will compute a Discrepancy term called Delta, which will tell us if the error locator
        # polynomial needs an update or not
        # (hence why it's called discrepancy: it tells us when we are getting off board from the correct value).

        # Init the polynomials
        # This is the main variable we want to fill, also called Sigma in other notations or more formally
        # the errors/errata locator polynomial.
        errors_locator = bytearray([1])
        # BM is an iterative algorithm, and we need the errata locator polynomial of the previous iteration in
        # order to update other necessary variables.
        old_errors_locator = bytearray([1])

        # Fix the syndromes shifting: when computing the syndromes, some implementations may prepend a 0 coefficient
        # for the lowest degree term (the constant). This is a case of syndromes shifting, thus the syndromes will be
        # bigger than the number of ecc symbols (I don't know what purpose serves this shifting). If that's the case,
        # then we need to account for the syndromes shifting when we use the syndromes such as inside BM, by skipping
        # those prepended coefficients. Another way to detect the shifting is to detect the 0 coefficients: by
        # definition, a syndromes does not contain any 0 coefficient (except if there are no errors/erasures,
        # in this case they are all 0). This however doesn't work with the modified Forney syndromes, which set to 0
        # the coefficients corresponding to erasures, leaving only the coefficients corresponding to errors.
        synd_shift = 0
        if len(syndromes) > self.redundant_characters:
            synd_shift = len(syndromes) - self.redundant_characters

        for i in range(self.redundant_characters):
            k = i + synd_shift

            # Compute the discrepancy Delta
            # Here is the close-to-the-books operation to compute the discrepancy  Delta: it's a simple polynomial
            # multiplication of error locator with the syndromes, and then we get the Kth element.
            # delta = gf_poly_mul(errors_locator[::-1], syndromes)[k] # theoretically it
            # should be polynomial_sum(syndromes[ ::-1], [1])[::-1] instead of just syndromes, but it seems
            # it's not absolutely necessary to correctly decode.
            # But this can be optimized: since we only need the Kth element, we don't need to compute the polynomial
            # multiplication for any other element but the Kth. Thus to optimize, we compute the polymul only at the
            # item we need, skipping the rest (avoiding a nested loop, thus we are linear time instead of quadratic).
            # This optimization is actually described in several figures of the book
            # "Algebraic codes for data transmission", Blahut, Richard E., 2003, Cambridge university press.
            delta = syndromes[k]
            for j in range(1, len(errors_locator)):
                # delta is also called discrepancy. Here we do a partial polynomial multiplication (ie, we compute the
                # polynomial multiplication only for the term of degree k). Should be equivalent to
                # brownanrs.polynomial.mul_at().
                delta ^= self.integer_multiplication(errors_locator[-(j + 1)], syndromes[k - j])


            # Shift polynomials to compute the next degree
            old_errors_locator = old_errors_locator + bytearray([0])

            # Iteratively estimate the errata locator and evaluator polynomials
            # Update only if there's a discrepancy
            if delta != 0:
                # Rule B (rule A is implicitly defined because rule A just says that we skip any modification
                # for this iteration)
                if len(old_errors_locator) > len(errors_locator):
                    # if 2*L <= k+erase_count: # equivalent to len(old_errors_locator) > len(errors_locator),
                    # as long as L is correctly computed Computing errata locator polynomial Sigma
                    new_loc = self.polynomial_and_integer_multiplication(old_errors_locator, delta)
                    # effectively we are doing errors_locator * 1/delta = errors_locator // delta
                    old_errors_locator = self.polynomial_and_integer_multiplication(errors_locator,
                                                                                    self.integer_inverse(delta))
                    errors_locator = new_loc
                    # Update the update flag L = k - L # the update flag L is tricky: in Blahut's schema,
                    # it's mandatory to use `L = k - L - erase_count` (and indeed in a previous draft of this
                    # function, if you forgot to do `- erase_count` it would lead to correcting only 2*(
                    # errors+erasures) <= (n-k) instead of 2*errors+erasures <= (n-k)), but in this latest draft,
                    # this will lead to a wrong decoding in some cases where it should correctly decode! Thus you
                    # should try with and without `- erase_count` to update L on your own implementation and see
                    # which one works OK without producing wrong decoding failures.

                # Update with the discrepancy
                errors_locator = self.polynomial_sum(errors_locator,
                                                     self.polynomial_and_integer_multiplication(old_errors_locator,
                                                                                                delta))

        # Check if the result is correct, that there's not too many errors to correct
        # drop leading 0s, else errs will not be of the correct size
        errors_locator = list(itertools.dropwhile(lambda x: x == 0, errors_locator))
        errs = len(errors_locator) - 1
        if errs * 2 > self.redundant_characters:
            raise Exception("Too many errors to correct")

        return errors_locator

    def find_errors_by_locator(self, error_locator, message_length):
        """Find the roots (ie, where evaluation = zero) of error polynomial by bruteforce trial, this is a sort of
        Chien's search """
        # message_length = length of whole codeword (message + ecc symbols)
        errs = len(error_locator) - 1
        err_pos = []
        # normally we should try all 2^8 possible values, but here we optimize to just check the interesting symbols
        for i in range(message_length):
            # It's a 0? Bingo, it's a root of the error locator polynomial, in other terms this is the location of
            # an error
            if self.calculate_polynomial_by_x(error_locator, self.polynomial_power(self.primitive_element, i)) == 0:
                err_pos.append(message_length - 1 - i)
        # Sanity check: the number of errors/errata positions found should be exactly the same as the length of the
        # errata locator polynomial
        if len(err_pos) != errs:
            raise Exception("Too many (or few) errors found by Chien Search for the errata locator polynomial!")
        return err_pos

    def correct_errors(self, message, syndromes, errors_positions, error_locator):
        """Forney algorithm, computes the values (error magnitude) to correct the input message."""
        msg = bytearray(message)
        # calculate errata locator polynomial to correct both errors and erasures (by combining the errors positions
        # given by the error locator polynomial found by BM with the erasures positions given by caller) need to
        # convert the positions to coefficients degrees for the errata locator algo to work (eg: instead of [0, 1,
        # 2] it will become [len(msg)-1, len(msg)-2, len(msg) -3])
        coef_pos = [len(msg) - 1 - p for p in errors_positions]

        # calculate errata evaluator polynomial (often called Omega or Gamma in academic papers)
        err_eval = self.calculate_error_evaluator(syndromes[::-1], error_locator)[::-1]


        # Second part of Chien search to get the error location polynomial x from the error positions in
        # errors_positions (the roots of the error locator polynomial, ie, where it evaluates to 0)
        polynomial_x = []  # will store the position of the errors
        for i in range(len(coef_pos)):
            temp_len = (self.galois_field_alphabet - 1) - coef_pos[i]
            polynomial_x.append(self.polynomial_power(self.primitive_element, -temp_len))

        # Forney algorithm: compute the magnitudes
        # will store the values that need to be corrected (substracted) to the message containing errors.
        # This is sometimes called the error magnitude polynomial.
        magnitude_polynomial = bytearray(len(msg))
        polynomial_x_length = len(polynomial_x)

        for i, x_i in enumerate(polynomial_x):
            xi_inverted = self.integer_inverse(x_i)

            # Compute the formal derivative of the error locator polynomial (see Blahut, Algebraic codes for data
            # transmission, pp 196-197). the formal derivative of the errata locator is used as the denominator of
            # the Forney Algorithm, which simply says that the ith error value is given by error_evaluator(
            # gf_inverse(x_i)) / error_locator_derivative(gf_inverse(x_i)). See Blahut, Algebraic codes for data
            # transmission, pp 196-197.
            err_loc_prime_tmp = []
            for j in range(polynomial_x_length):
                if j != i:
                    item = 1 ^ self.integer_multiplication(xi_inverted, polynomial_x[j])
                    err_loc_prime_tmp.append(item)
            # compute the product, which is the denominator of the Forney algorithm (errata locator derivative)
            err_loc_prime = 1
            for coefficient in err_loc_prime_tmp:
                err_loc_prime = self.integer_multiplication(err_loc_prime, coefficient)


            # Test if we could find the errata locator, else we raise an Exception (because else since we divide y by
            # err_loc_prime to compute the magnitude, we will get a ZeroDivisionError exception otherwise)
            if err_loc_prime == 0:
                raise Exception(
                    "Decoding failed: Forney algorithm could not properly detect where the errors are located (errata "
                    "locator prime is 0).")

            # Compute y (evaluation of the errata evaluator polynomial) This is a more faithful translation of the
            # theoretical equation contrary to the old forney method. Here it is exactly copy/pasted from the
            # included presentation decoding_rs.pdf: Yl = omega(Xl.inverse()) / prod(1 - Xj*Xl.inverse()) for j in
            # len(polynomial_x) (in the paper it's for j in s, but it's useless when len(polynomial_x) < s because we
            # compute neutral terms 1 for nothing, and wrong when correcting more than s erasures or erasures+errors
            # since it prevents computing all required terms). Thus here this method works with erasures too because
            # firstly we fixed the equation to be like the theoretical one (don't know why it was modified in
            # _old_forney(), if it's an optimization, it doesn't enhance anything), and secondly because we removed
            # the product bound on s, which prevented computing errors and erasures above the s=(n-k)//2 bound.
            # numerator of the Forney algorithm (errata evaluator evaluated)
            y = self.calculate_polynomial_by_x(err_eval[::-1], xi_inverted)
            # adjust to fcr parameter
            y = self.integer_multiplication(self.polynomial_power(x_i, 1), y)

            # Compute the magnitude magnitude value of the error, calculated by the Forney algorithm (an equation in
            # fact): dividing the errata evaluator with the errata locator derivative gives us the errata magnitude (
            # ie, value to repair) the ith symbol
            magnitude = self.integer_multiplication(y, self.integer_inverse(err_loc_prime))
            # store the magnitude for this error into the magnitude polynomial
            magnitude_polynomial[errors_positions[i]] = magnitude

            # Apply the correction of values to get our message corrected! (note that the ecc bytes also gets
            # corrected!)
        # (this isn't the Forney algorithm, we just apply the result of decoding here)
        # equivalent to Ci = Ri - Ei where Ci is the correct message, Ri the received (senseword) message, and Ei
        # the errata magnitudes (minus is replaced by XOR since it's equivalent in GF(2^p)).
        # So in fact here we substract from the received message the errors magnitude, which logically corrects
        # the value to what it should be.
        msg = self.polynomial_sum(msg, magnitude_polynomial)
        return msg

    def calculate_error_evaluator(self, syndromes, errors_locations):
        """Compute the error (or erasures if you supply sigma=erasures locator polynomial, or errata) evaluator
        polynomial Omega from the syndromes and the error/erasures/errata locator Sigma. Omega is already computed at
        the same time as Sigma inside the Berlekamp-Massey implemented above, but in case you modify Sigma,
        you can recompute Omega afterwards using this method, or just ensure that Omega computed by BM is correct
        given Sigma. """
        # Omega(x) = [ Synd(x) * Error_loc(x) ] mod x^(n-k+1)
        # first multiply syndromes * errata_locator, then do a polynomial division to truncate the polynomial to the
        # required length
        _, remainder = self.polynomials_division(
            self.polynomials_multiplication(syndromes, errors_locations), ([1] + [0] * (self.redundant_characters + 1))
        )
        return remainder


if __name__ == "__main__":
    rsc = ReedSolomonCoder(12)
   # rsc.encode(b'hel')
    print(rsc.encode(b'hel'))
    print(rsc.decode(b'hnl\xf0\x91'))