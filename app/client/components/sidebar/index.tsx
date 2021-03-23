import React from "react"
import Link from 'next/link'

export default function Sidebar() {

    return (
        <React.Fragment>
            <nav id="sidebarMenu" className="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse h-100">
                <div className="position-sticky pt-3">
                    <div className="mt-3 mb-5 ms-4 ps-2 fs-4">
                        MEL-AI
                    </div>
                    <ul className="nav flex-column">
                        <li className="nav-item mb-4">
                            <Link href={"/"} passHref>
                                <a className="nav-link active" aria-current="page">
                                    <span className="ms-3 text-black">
                                        Patients Overview
                                    </span>
                                </a>
                            </Link>
                        </li>
                    </ul>
                </div>
            </nav>
        </React.Fragment>
    )
}