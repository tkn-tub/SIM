classdef EpisodeCounter < handle
    % EpisodeCounter  Reference-semantics counter for deterministic episode
    %                 indexing in rlFunctionEnv reset callbacks.
    %
    %   Inherits from handle so that a single counter instance is shared
    %   across all calls to resetFunction, even when the EnvPars struct is
    %   captured by value inside an anonymous function. Calling next()
    %   increments the internal index and returns its new value; reset()
    %   sets it back to zero.
    %
    %   Typical usage:
    %       EnvPars.episode_counter = EpisodeCounter();
    %       % inside resetFunction:
    %       ep = EnvPars.episode_counter.next();
    
    properties
        idx = 0
    end
    methods
        function n = next(obj)
            obj.idx = obj.idx + 1;
            n = obj.idx;
        end
        function reset(obj)
            obj.idx = 0;
        end
    end
end