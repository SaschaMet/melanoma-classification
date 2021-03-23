import React from "react"
import { useQuery } from "@apollo/client";

import MediaDetail from "./mediaDetail"

import { MEDIA_BY_ENCOUNTER_ID } from "../../graphQl/queries/media"

export default function MediaList({ encounterId }) {


    const { loading, error, data } = useQuery(MEDIA_BY_ENCOUNTER_ID, {variables: { id: encounterId }});
    if (loading || error) return null

    const mediaIds = data.MediaByEncounterId.entries as String[]

    return (
        <div className="row">
            <div className="col-xs-12 col-md-8">
                {mediaIds.map(mediaId => <MediaDetail mediaId={mediaId} /> )}
            </div>
        </div>
    )
}