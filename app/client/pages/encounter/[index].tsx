import React from "react"
import { useQuery } from "@apollo/client";

import MediaList from "../../components/media/mediaList"

import { Encounter } from "../../types/fhir"
import { ENCOUNTER } from "../../graphQl/queries/encounter"

export default function EncounterDetail() {

    if(typeof window === 'undefined') return null

    const encounterId = window.location.pathname.split("/encounter/")[1]

    const { loading, error, data } = useQuery(ENCOUNTER, {variables: { id: encounterId }});

    if (loading) return <div className="col-4">Loading ...</div>;
    if (error) return <div className="col-4">An error occurred. Please try again.</div>;

    const encounter = data.Encounter as Encounter

    return (
        <div className="row">
            <div className="col-xs-12">
                <h1>Encounter</h1>
            </div>
            <div className="col-xs-12 mt-3">
                <p>
                    Status: {encounter.status}
                </p>
                <p>
                    Date: {encounter.meta.lastUpdated}
                </p>
            </div>
            <div className="col-xs-12 mt-5">
                <MediaList encounterId={encounterId} />
            </div>
        </div>
    )

}


