import steps.mpi
import time
if __name__ == '__main__':
    if steps.mpi.rank == 0:
        passed_tests=[]
        failed_tests=[]

    if steps.mpi.rank == 0:
        print "0. Surface Diffusion - Unbounded, point source (MPI TetOpSplit) with surface diffusion boundary:"
        start_time = time.time()
    import unbdiff2D_sdiffboundary
    if steps.mpi.rank == 0 and unbdiff2D_sdiffboundary.passed:
        print "PASSED"
        passed_tests.append("0. Surface Diffusion - Unbounded, point source (MPI TetOpSplit) with surface diffusion boundary")
        print "Time Cost: ", time.time() - start_time, " seconds."
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("0. Surface Diffusion - Unbounded, point source (MPI TetOpSplit) with surface diffusion boundary")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"

    if steps.mpi.rank == 0:
        print "1. Surface Diffusion - Unbounded, point source (MPI TetOpSplit):"
        start_time = time.time()
    import unbdiff2D
    if steps.mpi.rank == 0 and unbdiff2D.passed:
        print "PASSED"
        passed_tests.append("1. Surface Diffusion - Unbounded, point source (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds."
        print "NDiff: ", unbdiff2D.ndiff, " NReac: ", unbdiff2D.nreac, " Iterations: ", unbdiff2D.niteration, " Diff presentage: {:.2f}%".format(float(unbdiff2D.ndiff)/float(unbdiff2D.nreac + unbdiff2D.ndiff) * 100), " Average Diff per iteration: %i" % (unbdiff2D.ndiff / unbdiff2D.niteration), "\n"
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("1. Surface Diffusion - Unbounded, point source (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"

    if steps.mpi.rank == 0:
        print "2. Surface Diffusion - Unbounded, line source (MPI TetOpSplit):"
        start_time = time.time()
    import unbdiff2D_linesource_ring
    if steps.mpi.rank == 0 and unbdiff2D_linesource_ring.passed:
        print "PASSED"
        passed_tests.append("2. Surface Diffusion - Unbounded, line source (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds."
        print "NDiff: ", unbdiff2D_linesource_ring.ndiff, " NReac: ", unbdiff2D_linesource_ring.nreac, " Iterations: ", unbdiff2D_linesource_ring.niteration, " Diff presentage: {:.2f}%".format(float(unbdiff2D_linesource_ring.ndiff)/float(unbdiff2D_linesource_ring.nreac + unbdiff2D_linesource_ring.ndiff) * 100), " Average Diff per iteration: %i" % (unbdiff2D_linesource_ring.ndiff / unbdiff2D_linesource_ring.niteration), "\n"
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("2. Surface Diffusion - Unbounded, line source (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"
    
    if steps.mpi.rank == 0:
        print "3. Diffusion - Unbounded (MPI TetOpSplit):"
        start_time = time.time()
    import unbdiff
    if steps.mpi.rank == 0 and unbdiff.passed:
        print "PASSED"
        passed_tests.append("3. Diffusion - Unbounded (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds."
        print "NDiff: ", unbdiff.ndiff, " NReac: ", unbdiff.nreac, " Iterations: ", unbdiff.niteration, " Diff presentage: {:.2f}%".format(float(unbdiff.ndiff)/float(unbdiff.nreac + unbdiff.ndiff) * 100), " Average Diff per iteration: %i" % (unbdiff.ndiff / unbdiff.niteration), "\n"
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("3. Diffusion - Unbounded (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"

    if steps.mpi.rank == 0:
        print "4. Diffusion - Bounded (MPI TetOpSplit):"
        start_time = time.time()
    import bounddiff
    if steps.mpi.rank == 0 and bounddiff.passed:
        print "PASSED"
        passed_tests.append("4. Diffusion - Bounded (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds."
        print "NDiff: ", bounddiff.ndiff, " NReac: ", bounddiff.nreac, " Iterations: ", bounddiff.niteration, " Diff presentage: {:.2f}%".format(float(bounddiff.ndiff)/float(bounddiff.nreac + bounddiff.ndiff) * 100), " Average Diff per iteration: %i" % (bounddiff.ndiff / bounddiff.niteration), "\n"
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("4. Diffusion - Bounded (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"

    if steps.mpi.rank == 0:
        print "5. Diffusion - Clamped (MPI TetOpSplit):"
        start_time = time.time()
    import csd_clamp
    if steps.mpi.rank == 0 and csd_clamp.passed:
        print "PASSED"
        passed_tests.append("5. Diffusion - Clamped (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds."
        print "NDiff: ", csd_clamp.ndiff, " NReac: ", csd_clamp.nreac, " Iterations: ", csd_clamp.niteration, " Diff presentage: {:.2f}%".format(float(csd_clamp.ndiff)/float(csd_clamp.nreac + csd_clamp.ndiff) * 100), " Average Diff per iteration: %i" % (csd_clamp.ndiff / csd_clamp.niteration), "\n"
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("5. Diffusion - Clamped (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"

    if steps.mpi.rank == 0:
        print "6. Reaction-diffusion - Degradation-diffusion (MPI TetOpSplit):"
        start_time = time.time()
    import kisilevich
    if steps.mpi.rank == 0 and kisilevich.passed:
        print "PASSED"
        passed_tests.append("6. Reaction-diffusion - Degradation-diffusion (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds."
        print "NDiff: ", kisilevich.ndiff, " NReac: ", kisilevich.nreac, " Iterations: ", kisilevich.niteration, " Diff presentage: {:.2f}%".format(float(kisilevich.ndiff)/float(kisilevich.nreac + kisilevich.ndiff) * 100), " Average Diff per iteration: %i" % (kisilevich.ndiff / kisilevich.niteration), "R/D ratio: %i" % (float(kisilevich.ndiff) / kisilevich.nreac), "\n"
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("6. Reaction-diffusion - Degradation-diffusion (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"

    if steps.mpi.rank == 0:
        print "7. Reaction-diffusion - Production and second order degradation (MPI TetOpSplit):"
        start_time = time.time()
    import masteq_diff
    if steps.mpi.rank == 0 and masteq_diff.passed:
        print "PASSED"
        passed_tests.append("7. Reaction-diffusion - Production and second order degradation (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds."
        print "NDiff: ", masteq_diff.ndiff, " NReac: ", masteq_diff.nreac, " Iterations: ", masteq_diff.niteration, " Diff presentage: {:.2f}%".format(float(masteq_diff.ndiff)/float(masteq_diff.nreac + masteq_diff.ndiff) * 100), " Average Diff per iteration: %i" % (masteq_diff.ndiff / masteq_diff.niteration), "\n"
    elif steps.mpi.rank == 0:
        print "FAILED"
        failed_tests.append("7. Reaction-diffusion - Production and second order degradation (MPI TetOpSplit)")
        print "Time Cost: ", time.time() - start_time, " seconds.\n"

    if steps.mpi.rank == 0:
        print len(passed_tests), "tests passed:"
        for t in passed_tests:
            print t,
        print "\n", len(failed_tests), "tests failed:"
        for t in failed_tests:
            print t,
        print "\n"
