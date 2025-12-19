def decompose():
    """
            Calculate the eigen-decomposition of the given data using the specified method.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices 
            and n_maps is the number of brain maps.
        emodes : array-like
            The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of 
            eigenmodes.
        method : str, optional
            The method used for the eigen-decomposition, either 'orthogonal' or 'regress'. Default is
            'orthogonal'.
        mass : array-like, optional
            The mass matrix of shape (n_verts, n_verts) used for the eigen-decomposition when method
            is 'orthogonal'. If using EigenSolver, provide its self.mass. Default is None.
        return_norm_power : bool, optional
            If True, returns normalized power of each mode instead of beta coefficients. Default is
            False.
        check_orthonorm : bool, optional
            If True and mass is not None, checks that the eigenmodes are mass-orthonormal. Default 
            is True. See the check_orthonorm_modes function for details.

        Returns
        -------
        beta : numpy.ndarray
            The beta coefficients array of shape (n_modes, n_maps), obtained from the
            eigen-decomposition.
        norm_power : numpy.ndarray, optional
            The normalized power array of shape (n_modes, n_maps), obtained from the
            eigen-decomposition.

        Raises
        ------
        ValueError
            If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
            if an invalid method is specified, or if the `mass` matrix is not provided when
            required.
    """
    pass