#  Barnacle: A probabilistic model of RNA conformational space
#
#  Copyright (C) 2008 Jes Frellsen, Ida Moltke and Martin Thiim 
#
#  Barnacle is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Barnacle is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Barnacle.  If not, see <http://www.gnu.org/licenses/>.

from .model import make_model, angle_pos, hd_pos, num_angles
from .utils import make_data_and_mism, data_to_anglelist
from .InfEngineIOHMM import InfEngineIOHMM
from .SamplerIOHMM import SamplerFwBt
from .LikelihoodIOHMM import LikelohoodIOHMM_Fw
from .BarnacleExceptions import BarnacleException, BarnacleBoundException, BarnacleResampleException, BarnacleStateException
from .geometric_conversion import convert_from_dihedrals, residues_to_struct, ConversionException
from .NucleicAcid import rna_one_letter_codes

from Bio.PDB import PDBIO


class Barnacle:
    """
    A probabilistic model of RNA conformational space.

    The class is initialized with a nucleotide sequence. Subsequent
    calls to sample, will sample RNA structures that are compatible
    with the given sequence and that have good local conformational
    geometry.

    >>> model = Barnacle('ACCU')
    >>> model.sample()
    >>> model.save_structure('barnacle01.pdb')
    >>> model.sample()
    >>> model.save_structure('barnacle02.pdb')
    >>>
    """

    def __init__(self, nucleotide_sequence):
        """
        Class constructor. The class must be initialized with a
        nucleotide sequence.

        @param nucleotide_sequence: the nucleotide sequence
        @type nucleotide_sequence: string
        """

        # Check that nucleotide_sequence only contains valid nucleotides
        for nuc in nucleotide_sequence:
            if not nuc in rna_one_letter_codes:
                raise BarnacleException("Nucleotide sequence contains non-RNA nucleotide '%s'" % nuc)


        # Set basic variables
        self.nucs = nucleotide_sequence

        # Set up the DBN
        self.dbn = make_model()

        # Setup interface for writing PDB-files
        self.pdbio = PDBIO()

        # Make data and mismask
        self.data, self.mism = make_data_and_mism(len(self.nucs))
        self.structure = None
        self.anglelist = None
        self.prev = None
        
        # Setup the sampler
        self.sampler = SamplerFwBt(self.dbn, hd_pos)
        self.inf_engine = InfEngineIOHMM(self.dbn, self.sampler, self.data, self.mism, support_undo=True)
        self.sample_generator = self.inf_engine.get_sample_generator()

        # Setup the likelihood calculator
        self.ll_calc = LikelohoodIOHMM_Fw(self.dbn, hd_pos)

    def _save_prev(self):
        self.prev = (self.data.copy(), self.structure, self.anglelist)

    def _restore_prev(self):
        assert(self.prev!=None)
        (self.data, self.structure, self.anglelist) = self.prev
        self.prev = None

    def sample(self, start=None, end=None):
        """
        Sample new values for the dihedral angles.

        A subsequence of the angles can be sampled, conditioned on the
        remaining angles, if the parameters start and/or end are
        given.

        @param start: The first nucleotide position to sample angles
                      for (value None means 0).
        @type start: int

        @param end: The last nucleotide position to sample angles for
                    is end-1 (value None means len(nucleotide_sequence))
        @type end: int
        """

        # Save previous state
        self._save_prev()

        # Set default values of start and end
        if start == None:
            start = 0
        if end == None:
            end = len(self.nucs)

        # Check values of start and end
        if not 0<=start<end<=len(self.nucs):
            raise BarnacleBoundException("(start:end) out of bounds")
        if end-start != len(self.nucs) and not self.prev!=None:
            raise BarnacleResampleException("Cannot resample a subsequence, when no full sample has been drawn yet!")

        # Do the sampling and conversion
        self.inf_engine.set_start_end(start*num_angles, end*num_angles)

        while True:
            sample, slice_count = next(self.sample_generator)
            anglelist = data_to_anglelist(sample)

            try:
                residues = convert_from_dihedrals(anglelist, self.nucs, accept_no_solution=False)
            except ConversionException as e:
                self.inf_engine.undo()
                continue
            break

        # Save the sampled structure and angles
        self.data = sample
        self.anglelist = anglelist
        self.structure = residues_to_struct([('A', residues)], 'BARNACLE')
    
    def undo(self):
        """
        Undo the latest call to Barnacle.sample(), i.e. set the state
        of the class to what is was before sample was called. This
        method can only be called after Barnacle.sample() and only
        once for each call to Barnacle.sample().
        """
        if self.prev==None:
            raise BarnacleStateException("Barnacle.undo() is only allowed after a sample has been drawn. Multiple calls to undo are not supported.")

        if self.anglelist==None:
            raise BarnacleStateException("The first call to Barnacle.sample() cannot be followed by a call to Barnacle.undo().")

        self._restore_prev()
        self.inf_engine.undo()

    def get_log_likelihood(self):
        """
        Returns the log likelihood of the sampled angles (summed over all hidden states values).

        @return: the log likelihood of the sampled angles
        @rtype: float
        """
        if self.anglelist == None:
            raise BarnacleStateException("Barnacle.get_log_likelihood() can only be called after a sample has been drawn.")

        ll, slice_count = self.ll_calc.calc_ll(self.data, self.mism, set_data_in_dbn=False)
        return ll

    def get_angles(self):
        """
        Returns the sampled dihedral angles in the format. The angles
        are returned as a list of tuples of floats in the order:
        
        [(alpha, beta, gamma, delta, epsilon, zeta, chi), ..., (alpha, beta, gamma, delta, epsilon, zeta, chi)]

        Note that the angles are returned in radians.
        
        @return: the sampled angles
        @rtype: L{T{float}} object
        """

        return self.anglelist

    def get_structure(self):
        """
        Returns the atomic coordinates of the sampled structure as a
        Bio.PDB.Structure.
        
        @return: the sampled structure
        @rtype: Bio.PDB.Structure object
        """

        return self.structure

    def save_structure(self, filename="barnacle.pdb"):
        """
        Save the atomic coordinates of the sampled structure as a PDB file.

        @param filename: The filename of the PDB file to save.
        @type filename: string
        """

        pdbio = PDBIO()
        pdbio.set_structure(self.structure)
        pdbio.save(filename)

if __name__ == "__main__":
    # Test example
    from random import randint

    # The target sequence
    seq = "ACGU"

    # Initialize the model
    model = Barnacle(seq)

    # Draw first sample
    model.sample()

    for i in range(20):
        # Choose position to sample
        j = randint(0, len(seq)-1)

        # Sample position j
        model.sample(start=j, end=j+1)

        # Calculate and print likelihood
        ll = model.get_log_likelihood()
        print("structure_%03i.pdb: ll = %f" % (i, ll))
        
        # Save the structure
        model.save_structure("structure_%03d.pdb" % i)
